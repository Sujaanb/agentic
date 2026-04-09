import os
import re
import logging
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Load Gemini API key from environment or Streamlit secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate API key exists
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found. Set it in .env or environment variables.")
    st.info("Create a .env file from .env.example and add your API key.")
    st.stop()

try:
    # Initialize the Gemini LLM (Gemini 1.5 Pro model) for CrewAI
    llm = LLM(
        model="gemini/gemini-1.5-pro",
        api_key=GEMINI_API_KEY, 
        temperature=0.2  # low temperature for deterministic output
    )
except Exception as e:
    st.error(f"❌ Failed to initialize Gemini LLM: {str(e)}")
    logger.error(f"LLM initialization error: {str(e)}")
    st.stop()

# Define the Developer agent with a clear role and instructions
developer_agent = Agent(
    role="Web Developer",
    goal="Generate a complete HTML webpage with internal CSS styling based on the user's description.",
    backstory=(
        "You are an expert front-end developer. You will be given a description of a webpage.\n"
        "Your job is to output a single HTML file (with <html>, <head>, <body> tags) that implements the description.\n"
        "Include any required CSS in a <style> tag in the head for internal styling. Do NOT use external CSS files.\n"
        "Your response should contain only the HTML code, nothing else."
    ),
    llm=llm,
    verbose=False
)

# Define a Task that uses the agent to create HTML from a specification
generate_html_task = Task(
    description=(
        "Create a basic styled webpage based on the following requirements:\n"
        "\"\"\"{specification}\"\"\"\n\n"
        "Requirements:\n"
        " - Use HTML5 with a head and body section.\n"
        " - Add internal CSS styles in a <style> tag for styling as needed.\n"
        " - The content and design should match the description above.\n"
        "Output only the complete HTML code for the page."
    ),
    agent=developer_agent,
    expected_output="HTML code for the described webpage"
)

# Assemble the Crew with our single agent and task
crew = Crew(
    agents=[developer_agent],
    tasks=[generate_html_task],
    process=Process.sequential,
    verbose=False
)


def extract_html_from_response(response: str) -> str:
    """
    Extract valid HTML from LLM response.
    Handles cases where the LLM might include extra text around the HTML.
    """
    try:
        response_str = str(response).strip()
        
        # Try to extract content between <html> tags
        match = re.search(r'<html[^>]*>.*?</html>', response_str, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0)
        
        # If no html tags, check for body tags
        if '<body' in response_str.lower():
            # Extract from first opening tag to last closing tag
            first_tag = min(
                response_str.lower().find('<html'),
                response_str.lower().find('<!doctype'),
                response_str.lower().find('<body')
            )
            last_tag = response_str.lower().rfind('</html>')
            if last_tag > first_tag >= 0:
                return response_str[first_tag:last_tag + 7]
        
        # Fallback: wrap response in basic HTML
        if response_str:
            return f"<html><head><style>body {{ font-family: Arial, sans-serif; margin: 20px; }}</style></head><body>{response_str}</body></html>"
        
        return "<html><body><p>No content generated</p></body></html>"
    except Exception as e:
        logger.error(f"Error extracting HTML: {str(e)}")
        return "<html><body><p>Error generating HTML</p></body></html>"


# Streamlit UI
st.set_page_config(page_title="Agentic - HTML Generator", page_icon="🤖", layout="wide")
st.title("🤖 Agentic - AI HTML Generator")
st.write("Describe the webpage you want, and the AI developer will generate the HTML for you!")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **Agentic** uses CrewAI and Google Gemini to generate 
    professional HTML webpages from natural language descriptions.
    
    Simply describe what you want, and watch the magic happen! ✨
    """)
    st.divider()
    st.header("💡 Example Prompts")
    examples = [
        "Create a landing page for a tech startup with dark theme",
        "Make a contact form with validation",
        "Build a pricing table for a SaaS product",
        "Design a portfolio showcase for an artist"
    ]
    for i, example in enumerate(examples, 1):
        st.caption(f"{i}. {example}")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Display past conversation
if st.session_state.history:
    st.subheader("Chat History")
    for msg in st.session_state.history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write("**Generated HTML Preview:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Preview:**")
                    try:
                        st.components.v1.html(msg["content"], height=400, scrolling=True)
                    except Exception as e:
                        st.warning(f"Could not render preview: {str(e)}")
                with col2:
                    st.write("**Code:**")
                    st.code(msg["content"], language="html")

# Chat input for new prompt
st.divider()
user_input = st.chat_input("Describe the webpage you want to create...")

if user_input:
    # Display the user's message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate HTML with error handling
    with st.spinner("🔄 Generating your webpage..."):
        try:
            logger.info(f"User request: {user_input[:100]}...")
            result = crew.kickoff(inputs={"specification": user_input})
            html_output = extract_html_from_response(result)
            logger.info(f"HTML generated: {len(html_output)} characters")
            
            # Display the assistant message with the generated HTML
            with st.chat_message("assistant"):
                st.write("**Generated HTML Preview:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Preview:**")
                    try:
                        st.components.v1.html(html_output, height=400, scrolling=True)
                    except Exception as e:
                        st.warning(f"Could not render preview: {str(e)}")
                        st.info("But the HTML code below should work!")
                
                with col2:
                    st.write("**Code:**")
                    st.code(html_output, language="html")
                    
                    # Add copy button
                    st.button(
                        "📋 Copy HTML Code",
                        key=f"copy_{len(st.session_state.history)}",
                        help="Copy the HTML code to clipboard (use browser's copy feature)"
                    )
            
            # Save conversation history
            st.session_state.history.append({"role": "user", "content": user_input})
            st.session_state.history.append({"role": "assistant", "content": html_output})
            
            st.success("✅ HTML generated successfully!")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating HTML: {error_msg}")
            st.error(f"❌ Error generating HTML: {error_msg}")
            st.info("Tips to fix the error:")
            st.write("1. Check your GEMINI_API_KEY is valid")
            st.write("2. Verify you have sufficient API quota")
            st.write("3. Try a simpler, more specific prompt")
            st.write("4. Check your internet connection")
