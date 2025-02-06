import streamlit as st
from typing import Optional, List
from langchain.llms.base import LLM
from groq import Groq
from tavily import TavilyClient
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

if not GROQ_API_KEY or not TAVILY_API_KEY:
    raise ValueError("Missing required API keys in environment variables")

client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Initialize session state for caching
if 'search_cache' not in st.session_state:
    st.session_state.search_cache = {}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def search_courses(topic: str) -> list:
    """Search for online courses using Tavily with caching."""
    # Check cache first
    cache_key = f"search_{topic.lower()}"
    if cache_key in st.session_state.search_cache:
        return st.session_state.search_cache[cache_key]

    try:
        # Search specifically for each platform
        platforms = ["Udemy", "Coursera", "YouTube"]
        all_results = []

        for platform in platforms:
            search_query = f"best {platform} courses for learning {topic}"
            search_result = tavily_client.search(
                search_query,
                search_depth="advanced",
                max_results=3
            )

            # Process results for each platform
            for result in search_result['results']:
                platform_tag = f"[{platform}]"
                # Extract potential rating from title or description
                rating = "N/A"
                if "rating" in result['title'].lower() or "stars" in result['title'].lower():
                    rating = "⭐⭐⭐⭐⭐"

                all_results.append({
                    'title': f"{platform_tag} {result['title']}",
                    'url': result['url'],
                    'description': result.get('content', 'No description available'),
                    'platform': platform,
                    'rating': rating
                })

        # Cache the results
        st.session_state.search_cache[cache_key] = all_results
        return all_results

    except Exception as e:
        st.error(f"Error searching for courses: {str(e)}")
        time.sleep(1)  # Rate limit handling
        return []

# Modified Groq LLM Wrapper
class GroqLLM(LLM):
    model: str = "llama-3.3-70b-specdec"
    temperature: int = 1
    max_tokens: int = 5096

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error calling Groq API: {str(e)}")
            raise

TUTOR_PROMPT = """I want to learn {topic} from the world's best professional-YOU. You are the ultimate expert, the top authority in this field, and the best tutor anyone could ever learn from. No one can match your knowledge and expertise.

Course Parameters:
- Duration: {duration} weeks
- Learning Style: {learning_style}
- Current Proficiency: {proficiency}

Teach me everything from basic to advanced, covering every minute detail in a structured and progressive manner, optimized for my {duration}-week timeline. Adapt the content to my {learning_style} learning style and {proficiency} proficiency level.

For each week, create a structured lesson plan that includes:
- **Weekly Goal**: A clear objective for what I should achieve by the end of the week.
- **Topics to Cover**:  List the specific topics within {topic} that should be learned this week.
- **Learning Resources**: Suggest specific resources like YouTube videos, Coursera courses, Udemy courses, documentation, or blog posts. Provide direct links if possible.
- **Exercises**: Include 2-3 exercises to solidify my understanding of the week's topics. These should be practical exercises, coding challenges (if applicable to the topic), or questions that encourage critical thinking and application of the learned concepts.
- **Cheat Sheet**: Briefly summarize the key terms, definitions, or formulas from that week's content in bullet points.

Start with foundational concepts in Week 1 and progressively move to more advanced topics in subsequent weeks. Ensure the plan is suitable for a {duration}-week course and tailored to my {learning_style} learning style and {proficiency} proficiency level.

Format each week's plan clearly with headings and bullet points for easy readability.

Let's begin creating the {duration}-week learning plan for {topic}."""


def display_course_card(course):
    """Display a formatted course card."""
    with st.expander(course['title']):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Platform:** {course['platform']}")
            st.markdown(f"**Rating:** {course['rating']}")
        with col2:
            st.markdown(f"[Go to Course]({course['url']})")
        st.markdown("---")
        st.markdown(course['description'])

def main():
    st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓")

    # Add custom CSS
    st.markdown("""
        <style>
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .learning-plan-section {
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            margin-bottom: 20px;
            background-color: #f9f9f9; /* Light grey background */
        }
        .learning-plan-section h3 {
            color: #333; /* Dark heading color */
        }
        .learning-plan-section ul {
            list-style-type: disc;
            margin-left: 20px;
        }
        .learning-plan-section li {
            margin-bottom: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("EduGenie 🎓")
    st.write("I'll create a personalized lesson plan with exercises and find relevant courses for you!")

    # Topic Input
    topic = st.text_input("What topic would you like to learn?")

    # Course Parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        duration = st.selectbox(
            "Course Duration",
            options=[1, 2, 4, 8, 12, 16, 24],
            format_func=lambda x: f"{x} weeks"
        )

    with col2:
        learning_style = st.selectbox(
            "Learning Style",
            options=[
                "Visual",
                "Auditory",
                "Reading/Writing",
                "Mixed"
            ]
        )

    with col3:
        proficiency = st.selectbox(
            "Current Proficiency",
            options=[
                "Complete Beginner",
                "Some Basic Knowledge",
                "Intermediate",
                "Advanced"
            ]
        )

    # Generate button
    generate_button = st.button("Generate Learning Plan 🚀", type="primary")

    if topic and generate_button:  # Only process when button is clicked
        with st.spinner("Creating your personalized learning plan..."):
            try:
                # Create progress bar
                progress_bar = st.progress(0)

                # Get learning plan
                progress_bar.progress(30)
                groq_llm = GroqLLM()
                prompt = TUTOR_PROMPT.format(
                    topic=topic,
                    duration=duration,
                    learning_style=learning_style,
                    proficiency=proficiency
                )
                learning_plan = groq_llm._call(prompt)

                progress_bar.progress(60)

                # Display learning plan in a more structured format
                st.success("Your Personalized Learning Plan")

                # --- Improved Learning Plan Display ---
                learning_plan_sections = learning_plan.split("Week ") # Assuming LLM formats with "Week 1:", "Week 2:", etc.

                for section in learning_plan_sections[1:]: # Skip the first empty split
                    week_num = section.split(":")[0] # Extract week number
                    week_content = ":".join(section.split(":")[1:]).strip() # Get content after week number
                    if week_content: # Avoid empty sections
                        with st.container():
                            st.markdown(f"<div class='learning-plan-section'><h3>Week {week_num}</h3>{week_content}</div>", unsafe_allow_html=True)
                # --- End Improved Learning Plan Display ---


                progress_bar.progress(80)

                # Search and display relevant courses
                st.write("---")
                st.subheader("📚 Recommended Online Courses")

                courses = search_courses(topic)
                if courses:
                    # Group courses by platform
                    for platform in ["Udemy", "Coursera", "YouTube"]:
                        platform_courses = [c for c in courses if c['platform'] == platform]
                        if platform_courses:
                            st.markdown(f"### {platform} Courses")
                            for course in platform_courses:
                                display_course_card(course)
                else:
                    st.info("No courses found at the moment. Try refining your search.")

                progress_bar.progress(100)

                # Add feedback option with improved UI
                st.write("---")
                with st.expander("💭 Provide Feedback"):
                    feedback = st.text_area("How can we improve this learning plan?")
                    if feedback:
                        st.success("Thank you for your valuable feedback! 🙏")

            except Exception as e:
                st.error(f"Error creating learning plan: {str(e)}")

if __name__ == "__main__":
    main()
