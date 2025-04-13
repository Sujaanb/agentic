import os
from crewai import Agent, Task, Crew, Process, LLM

# Load Gemini API key from environment or Streamlit secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # or st.secrets["GEMINI_API_KEY"] in a Streamlit app

# Initialize the Gemini LLM (Gemini 1.5 Pro model) for CrewAI
llm = LLM(
    model="gemini/gemini-1.5-pro",    # using Gemini Pro model (1.5)
    api_key=GEMINI_API_KEY, 
    temperature=0.2                  # low temperature for deterministic output (adjust as needed)
)
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
    verbose=False  # turn off verbose logging for cleaner output
)

# Define a Task that uses the agent to create HTML from a specification
generate_html_task = Task(
    description=(
        "Create a basic styled webpage based on the following requirements:\n"
        "\"\"\"{specification}\"\"\"\n\n"  # {specification} will be filled with the user prompt
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
    process=Process.sequential,  # tasks will run sequentially&#8203;:contentReference[oaicite:3]{index=3}
    verbose=False
)


import streamlit as st
st.title("HTML Generator Chatbot (CrewAI + Gemini)")
st.write("Describe the webpage you want, and the AI developer will generate the HTML for you:")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Display past conversation (user prompts and bot responses)
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        # Display the assistant's HTML output inside an expandable container
        with st.chat_message("assistant"):
            st.write("**Generated HTML Preview:**")
            st.components.v1.html(msg["content"], height=400, scrolling=True)
            # Also show the raw HTML code (optional):
            # st.code(msg["content"], language="html")
            
# Chat input for new prompt
user_input = st.chat_input("Enter a description for your webpage...")
if user_input:
    # Display the user's message
    st.chat_message("user").write(user_input)
    # Run the CrewAI agent to get HTML output
    result = crew.kickoff(inputs={"specification": user_input})
    html_output = str(result)  # convert the result to string, if it's not already
    
    # Display the assistant message with the generated HTML
    with st.chat_message("assistant"):
        st.write("**Generated HTML Preview:**")
        st.components.v1.html(html_output, height=400, scrolling=True)
        # st.code(html_output, language="html")  # (optional) show code
    
    # Save conversation history
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": html_output})
