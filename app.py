import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Custom CSS for enhanced UI and header positioning
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">

    <style>
    body {
        background-color: #f0f2f5;
    }
    .header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1000;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        align-items : center;
        color: #fff;
        padding: 20px;
        background: linear-gradient(135deg, #89CFF0, #A7C7E7);
        border-radius: 10px;
    }
    @keyframes fadeInDown {
        0% {
            opacity: 0;
            transform: translateY(-20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .search-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 10px;
        background-color: #fff;
        position: fixed;
        bottom: 0;
        width: 100%;
        box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.1);
    }
    .search-container input {
        flex: 1;
        padding: 10px;
        font-size: 1.2em;
        border: 2px solid #89CFF0;
        border-radius: 10px;
    }
    .settings {
        color: #89CFF0;
        font-size: 1.5em;
        margin-bottom: 20px;
        animation: fadeInLeft 1s ease-in-out;
    }
    @keyframes fadeInLeft {
        0% {
            opacity: 0;
            transform: translateX(-20px);
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }
    .chat-message {
        padding: 10px;
        margin: 10px 0;
        border-radius: 10px;
        animation: bounceIn 1s ease-in-out;
    }
    .chat-message.user {
        background-color: #89CFF0;
        color: white;
    }
    .chat-message.assistant {
        background-color: #f0f2f5;
        color: black;
        border: 2px solid #89CFF0;
    }
    @keyframes bounceIn {
        0%, 20%, 40%, 60%, 80%, 100% {
            transition-timing-function: cubic-bezier(0.215, 0.61, 0.355, 1);
        }
        0% {
            opacity: 0;
            transform: scale3d(0.3, 0.3, 0.3);
        }
        20% {
            transform: scale3d(1.1, 1.1, 1.1);
        }
        40% {
            transform: scale3d(0.9, 0.9, 0.9);
        }
        60% {
            opacity: 1;
            transform: scale3d(1.03, 1.03, 1.03);
        }
        80% {
            transform: scale3d(0.97, 0.97, 0.97);
        }
        100% {
            opacity: 1;
            transform: scale3d(1, 1, 1);
        }
    }
    </style>
    """, unsafe_allow_html=True)

# JavaScript to manage header visibility during search
st.markdown("""
    <script>
    # 
    </script>
    """, unsafe_allow_html=True)

# Render the header
st.markdown('<div class="header" id="header">.</div>', unsafe_allow_html=True)

# Sidebar for settings
st.sidebar.markdown("""
    <div class="settings">
        <i class="fas fa-cog"></i> Search Bot Settings
    </div>
    """, unsafe_allow_html=True)

api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hey! I'm ready to help you with your web search. What would you like to know?"}
    ]

for msg in st.session_state.messages:
    msg_class = "user" if msg["role"] == "user" else "assistant"
    st.markdown(f'<div class="chat-message {msg_class}">{msg["content"]}</div>', unsafe_allow_html=True)

# Define a function to handle search execution
def run_search():
    # Hide the header during search
    st.markdown('<script>hideHeader();</script>', unsafe_allow_html=True)

    if st.session_state.prompt:
        st.session_state.messages.append({"role": "user", "content": st.session_state.prompt})
        st.markdown(f'<div class="chat-message user">{st.session_state.prompt}</div>', unsafe_allow_html=True)

        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
        tools = [search, arxiv, wiki]

        search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

        with st.spinner('Searching...'):
            try:
                response = search_agent.run(st.session_state.messages)
                st.session_state.messages.append({'role': 'assistant', "content": response})
                st.markdown(f'<div class="chat-message assistant">{response}</div>', unsafe_allow_html=True)
            except ValueError as e:
                st.session_state.messages.append({'role': 'assistant', "content": str(e)})
                st.markdown(f'<div class="chat-message assistant">An error occurred: {str(e)}</div>', unsafe_allow_html=True)

        # Show the header between the search results
        st.markdown('<script>showHeader();</script>', unsafe_allow_html=True)

        # Clear the search bar after submission
        st.session_state.prompt = ""

# Add the search input field at the bottom
search_container = st.container()
with search_container:
    prompt = st.text_input("Message ChatBOT...", placeholder="Type your message here", on_change=run_search, key="prompt")
