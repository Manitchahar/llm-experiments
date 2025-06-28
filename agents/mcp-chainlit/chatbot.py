import chainlit as cl
import os
import yaml
import asyncio
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from mcp import ClientSession
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Load configuration from YAML file
def load_config():
    """Load configuration from config.yaml file"""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        # Fallback to default values if config file doesn't exist
        return {
            'llm': {
                'model_name': 'meta-llama/llama-4-scout-17b-16e-instruct',
                'temperature': 0.7
            },
            'agent': {
                'memory_window': 10,
                'tool_timeout': 30
            }
        }
    except yaml.YAMLError as e:
        print(f"Error loading config.yaml: {e}")
        # Return default values on error
        return {
            'llm': {
                'model_name': 'meta-llama/llama-4-scout-17b-16e-instruct',
                'temperature': 0.7
            },
            'agent': {
                'memory_window': 10,
                'tool_timeout': 30
            }
        }

# Load configuration
config = load_config()

def extract_tool_command(text: str):
    """
    Robust tool command extraction with multiple fallback patterns
    Returns: (tool_name, json_params) or (None, None) if no match
    """
    import re
    
    # Pattern 1: Most robust - handles tool names with hyphens, dots, underscores
    # and properly balanced JSON with nested braces
    patterns = [
        # Pattern 1: Tool names with special chars + balanced braces
        r'run\s+tool\s+([\w\-\.\_]+)\s+with\s+(\{(?:[^{}]|{[^{}]*})*\})',
        
        # Pattern 2: More permissive JSON capture (original improved)
        r'run\s+tool\s+([\w\-\.\_]+)\s+with\s+(\{.*?\})',
        
        # Pattern 3: Simple word-based tool names (backwards compatibility)
        r'run\s+tool\s+(\w+)\s+with\s+(\{[^}]*\})',
        
        # Pattern 4: Last resort - capture everything after 'with'
        r'run\s+tool\s+([\w\-\.\_]+)\s+with\s+(.*?)(?:\n|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            tool_name = match.group(1).strip()
            json_str = match.group(2).strip()
            
            # Validate that we have a reasonable tool name
            if len(tool_name) > 0 and len(tool_name) < 100:
                # Try to validate JSON-like structure
                if json_str.startswith('{') and json_str.endswith('}'):
                    return tool_name, json_str
                elif json_str.startswith('{'):
                    # Try to find the closing brace
                    brace_count = 0
                    end_pos = 0
                    for i, char in enumerate(json_str):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    if end_pos > 0:
                        json_str = json_str[:end_pos]
                        return tool_name, json_str
    
    return None, None

def format_tool_output(output: str, tool_name: str, max_length: int = None) -> str:
    """
    Smart formatting for tool outputs with length management and content detection
    """
    if max_length is None:
        max_length = config['agent']['max_output_length']
    
    truncate_threshold = config['agent']['truncate_threshold']
    
    # Handle None or empty output
    if not output:
        return "*No output returned*"
    
    # Convert to string if not already
    output_str = str(output).strip()
    
    if not output_str:
        return "*Empty output*"
    
    # Detect and format JSON
    if output_str.startswith('{') or output_str.startswith('['):
        try:
            import json
            parsed = json.loads(output_str)
            formatted_json = json.dumps(parsed, indent=2, ensure_ascii=False)
            output_str = formatted_json
        except json.JSONDecodeError:
            pass  # Not valid JSON, treat as regular text
    
    # Handle very long outputs
    if len(output_str) > max_length:
        if len(output_str) > truncate_threshold:
            truncated = output_str[:truncate_threshold]
            # Try to break at a natural boundary (newline, sentence, etc.)
            for break_char in ['\n\n', '\n', '. ', '! ', '? ']:
                break_pos = truncated.rfind(break_char)
                if break_pos > truncate_threshold * 0.7:  # At least 70% of threshold
                    truncated = truncated[:break_pos + len(break_char)]
                    break
            
            lines_remaining = output_str[len(truncated):].count('\n')
            chars_remaining = len(output_str) - len(truncated)
            
            truncation_note = f"\n\n*[Output truncated - {chars_remaining} more characters"
            if lines_remaining > 0:
                truncation_note += f", {lines_remaining} more lines"
            truncation_note += "]*"
            
            output_str = truncated + truncation_note
    
    # Sanitize output for markdown safety
    # Escape potential markdown that could break formatting
    output_str = output_str.replace('```', '\\`\\`\\`')
    
    # Handle special content types
    if tool_name in ['screenshot', 'image', 'photo']:
        if 'base64' in output_str.lower() or len(output_str) > 1000:
            return "*[Image/Binary data returned - content not displayed]*"
    
    return output_str

def extract_mcp_result_content(result):
    """
    Robust extraction of content from MCP tool results
    """
    try:
        # Handle different MCP response structures
        if hasattr(result, 'content') and result.content:
            if isinstance(result.content, list) and len(result.content) > 0:
                first_content = result.content[0]
                if hasattr(first_content, 'text'):
                    return first_content.text
                elif hasattr(first_content, 'content'):
                    return first_content.content
                else:
                    return str(first_content)
            else:
                return str(result.content)
        
        # Fallback to string representation
        elif hasattr(result, 'text'):
            return result.text
        
        # Last resort
        else:
            result_str = str(result)
            # Don't return object representations like <Object at 0x...>
            if 'object at 0x' in result_str:
                return "*[Complex object returned - cannot display content]*"
            return result_str
    
    except Exception as e:
        return f"*[Error extracting result content: {str(e)}]*"

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Called when an MCP connection is established"""
    try:
        # List available tools from the MCP server
        result = await session.list_tools()
        
        # Process tool metadata
        tools = [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema,
        } for t in result.tools]
        
        # Store tools for later use
        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = tools
        cl.user_session.set("mcp_tools", mcp_tools)
        
        # Send a message to the user about the successful connection
        await cl.Message(
            content=f"✅ MCP connection '{connection.name}' established successfully! "
                   f"Available tools: {', '.join([t['name'] for t in tools])}"
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ Error connecting to MCP server '{connection.name}': {str(e)}"
        ).send()

@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Called when an MCP connection is terminated"""
    # Clean up tools from this connection
    mcp_tools = cl.user_session.get("mcp_tools", {})
    if name in mcp_tools:
        del mcp_tools[name]
        cl.user_session.set("mcp_tools", mcp_tools)
    
    await cl.Message(
        content=f"🔌 MCP connection '{name}' disconnected"
    ).send()

@cl.step(type="tool")
async def call_mcp_tool(tool_name: str, tool_input: dict, mcp_name: str):
    """Execute an MCP tool with timeout protection"""
    try:
        # Get the MCP session
        mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
        
        if not mcp_session:
            return {"error": f"MCP connection '{mcp_name}' not found"}
        
        # Call the tool with timeout protection
        timeout = config['agent']['tool_timeout']
        result = await asyncio.wait_for(
            mcp_session.call_tool(tool_name, tool_input), 
            timeout=timeout
        )
        return result
        
    except asyncio.TimeoutError:
        return {"error": f"Tool '{tool_name}' timed out after {config['agent']['tool_timeout']} seconds"}
    except Exception as e:
        return {"error": f"Error calling tool '{tool_name}': {str(e)}"}

@cl.on_chat_start
async def main():
    """Initialize the chatbot when a new chat session starts."""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            await cl.Message(content="Error: GROQ_API_KEY not found in environment variables.").send()
            return

        # Initialize the Groq LLM using configuration
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=config['llm']['model_name'],
            temperature=config['llm']['temperature'],
            streaming=True
        )

        # Initialize conversation memory using configuration
        memory = ConversationBufferWindowMemory(
            k=config['agent']['memory_window'],  # Keep the last N messages from config
            memory_key="chat_history",
            return_messages=True
        )

        # Store in user session
        cl.user_session.set("llm", llm)
        cl.user_session.set("memory", memory)

        # Send welcome message
        welcome_msg = """👋 Hello! I'm your AI assistant with MCP integration.

🔧 **What I can do:**
- Have conversations with memory of our chat history
- Connect to MCP servers to run tools like PowerShell commands
- Help you with various tasks using available tools

📡 **To connect MCP servers:**
- Click the "MCP" button in the chat interface to manage connections.

Let me know how I can help you today!"""
        await cl.Message(content=welcome_msg).send()

    except Exception as e:
        await cl.Message(content=f"Failed to initialize chatbot: {str(e)}").send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""
    try:
        llm = cl.user_session.get("llm")
        memory = cl.user_session.get("memory")
        
        if not llm or not memory:
            await cl.Message(content="Error: Chatbot is not initialized.").send()
            return

        # Get available MCP tools
        mcp_tools = cl.user_session.get("mcp_tools", {})
        
        # Build context about available tools
        tool_context = ""
        if mcp_tools:
            tool_context = "\n\nAvailable MCP Tools:\n"
            for mcp_name, tools in mcp_tools.items():
                tool_context += f"\nConnection '{mcp_name}':\n"
                for tool in tools:
                    tool_context += f"- {tool['name']}: {tool['description']}\n"
            
            tool_context += "\nIf the user asks for something that can be done with these tools, offer to use them. To use a tool, ask me to 'run tool [tool_name] with [parameters]'."

        # Add user message to memory
        memory.chat_memory.add_message(HumanMessage(content=message.content))

        # Get chat history
        chat_history = memory.chat_memory.messages

        # Create system message with tool context
        system_prompt = f"""You are a helpful AI assistant with access to MCP tools that you can execute automatically. You have access to conversation history and can see a list of available MCP tools.

When a user asks for something that can be done with available tools, you should:
1. Determine the appropriate tool and parameters needed
2. Execute the tool automatically by responding with the exact format: `run tool <tool_name> with {{"parameter_name": "value", ...}}`
3. Do not ask for permission - execute the tool directly

For example, if a user asks "what are the system specs" or "list the files", immediately respond with the appropriate tool command like:
`run tool run_powershell with {{\"code\": \"Get-ComputerInfo | Select-Object WindowsProductName, TotalPhysicalMemory, CsProcessors\"}}`

Available MCP Tools:{tool_context}
"""

        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add chat history (using config for consistency)
        for msg in chat_history[-config['agent']['memory_window']:]:  # Keep last N messages for context
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})

        # First, check if the message is a tool execution command
        if message.content.strip().startswith("run tool"):
            await handle_tool_request(message.content, mcp_tools)
            return  # Stop processing after handling the tool command

        # If not a tool command, proceed with the conversational response using native Groq streaming
        msg = cl.Message(content="")
        full_response = ""
        
        # Use Groq's native astream for token-by-token streaming
        async for chunk in llm.astream(messages):
            if chunk.content:
                await msg.stream_token(chunk.content)
                full_response += chunk.content
        
        await msg.send()
        
        # Check if the AI's response contains a tool command using robust extraction
        if "run tool" in full_response.lower():
            # Use robust tool command extraction
            tool_name, params_str = extract_tool_command(full_response)
            
            if tool_name and params_str:
                # --- VALIDATION: Check if the tool exists ---
                is_valid_tool = False
                for mcp_name, tools in mcp_tools.items():
                    if any(t['name'] == tool_name for t in tools):
                        is_valid_tool = True
                        break
                
                if not is_valid_tool:
                    await cl.Message(content=f"⚠️ The AI tried to use a tool named '{tool_name}', but it doesn't exist in any connected MCP server. Let me try to help without using tools.").send()
                    # Add AI response to memory and continue with regular conversation
                    memory.chat_memory.add_message(AIMessage(content=full_response))
                    return
                
                tool_command = f"run tool {tool_name} with {params_str}"
                # Add AI response to memory first
                memory.chat_memory.add_message(AIMessage(content=f"I'll execute that command for you."))
                # Send detailed execution message with full command
                await cl.Message(content=f"🔧 **Executing:**\n```\n{tool_command}\n```").send()
                # Execute the tool
                await handle_tool_request(tool_command, mcp_tools)
                return
        
        # Add AI response to memory
        memory.chat_memory.add_message(AIMessage(content=full_response))

    except Exception as e:
        await cl.Message(content=f"Sorry, I encountered an error: {str(e)}").send()

async def handle_tool_request(user_input: str, mcp_tools: dict):
    """Handle generic tool execution requests with robust parsing and formatting"""
    import json

    # Use robust tool command extraction
    tool_name, params_str = extract_tool_command(user_input)

    if not tool_name or not params_str:
        await cl.Message(content="❌ Invalid command format. Please use: `run tool <tool_name> with {<parameters>}`").send()
        return

    try:
        # Parse the JSON parameters
        tool_input = json.loads(params_str)
    except json.JSONDecodeError as e:
        await cl.Message(content=f"❌ Invalid JSON parameters for tool '{tool_name}': {str(e)}\n\nPlease check your JSON syntax.").send()
        return

    # Find the MCP connection that provides the requested tool
    mcp_name = None
    for conn_name, tools in mcp_tools.items():
        if any(tool['name'] == tool_name for tool in tools):
            mcp_name = conn_name
            break
    
    if not mcp_name:
        await cl.Message(content=f"❌ Tool '{tool_name}' not found in any active MCP connection.").send()
        return

    # Execute the tool
    try:
        result = await call_mcp_tool(tool_name, tool_input, mcp_name)
        
        if isinstance(result, dict) and "error" in result:
            await cl.Message(content=f"❌ **Error:**\n```\n{result['error']}\n```").send()
        else:
            # Extract and format the output using robust methods
            raw_output = extract_mcp_result_content(result)
            formatted_output = format_tool_output(raw_output, tool_name)
            
            # Show detailed execution info with smart formatting
            execution_details = f"""✅ **Tool Executed Successfully**

**Tool:** `{tool_name}`
**Parameters:** 
```json
{json.dumps(tool_input, indent=2)}
```

**Output:**
```
{formatted_output}
```"""
            await cl.Message(content=execution_details).send()
            
            # Get LLM and memory from user session for follow-up interpretation
            llm = cl.user_session.get("llm")
            memory = cl.user_session.get("memory")
            
            if llm and memory:
                # Create a follow-up prompt for the AI to interpret the results
                interpretation_prompt = f"""
The tool execution has completed successfully. Please analyze and summarize the output for the user in a helpful way.

**Tool executed:** {tool_name}
**Parameters:** {json.dumps(tool_input, indent=2)}
**Raw output:** 
{raw_output}

Please provide:
1. A clear summary of what was found/accomplished
2. Key highlights or important details
3. Any insights or patterns you notice
4. Suggest follow-up actions if relevant

Be conversational and helpful. Don't just repeat the raw output - interpret it meaningfully for the user.
"""
                
                try:
                    # Get AI interpretation of the results
                    interpretation_response = await llm.ainvoke([
                        {"role": "system", "content": "You are a helpful AI assistant that interprets tool execution results and provides meaningful summaries to users."},
                        {"role": "user", "content": interpretation_prompt}
                    ])
                    
                    # Add the interpretation to memory and send to user
                    memory.chat_memory.add_message(AIMessage(content=interpretation_response.content))
                    await cl.Message(content=f"🤖 **Analysis:**\n\n{interpretation_response.content}").send()
                    
                except Exception as interpretation_error:
                    await cl.Message(content=f"⚠️ Tool executed successfully, but couldn't analyze results: {str(interpretation_error)}").send()
            
    except Exception as e:
        await cl.Message(content=f"❌ **Execution Error:**\n```\n{str(e)}\n```").send()
