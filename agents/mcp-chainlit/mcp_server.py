from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("powershell-integration")

# Define the command to run PowerShell code
@mcp.tool()
def run_powershell(code: str) -> str:
    """Runs PowerShell code and returns the output."""
    import subprocess

    # Run the PowerShell command
    process = subprocess.Popen(
        ["powershell", "-Command", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Get the output and error messages
    output, error = process.communicate()

    if process.returncode != 0:
        return f"Error: {error}"

    return output

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()