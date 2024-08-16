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

# Custom CSS for enhanced UI
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">

    <style>
    body {
        background-color: #f0f2f5;
    }
    .header {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        color: #fff;
        padding: 20px;
        margin-bottom: 40px;
        background: linear-gradient(135deg, #ff7e5f, #feb47b);
        border-radius: 10px;
        animation: fadeInDown 1s ease-in-out;
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
        margin-top: 20px;
        margin-bottom: 20px;
        animation: slideInUp 1s ease-in-out;
    }
    .search-container input {
        flex: 1;
        padding: 10px;
        font-size: 1.2em;
        border: 2px solid #ff7e5f;
        border-radius: 10px 0 0 10px;
    }
    .search-container button {
        padding: 10px;
        font-size: 1.2em;
        background-color: #ff7e5f;
        color: white;
        border: none;
        cursor: pointer;
        border-radius: 0 10px 10px 0;
        transition: background-color 0.3s ease;
    }
    .search-container button:hover {
        background-color: #feb47b;
    }
    @keyframes slideInUp {
        0% {
            opacity: 0;
            transform: translateY(20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .settings {
        color: #ff7e5f;
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
        background-color: #ff7e5f;
        color: white;
    }
    .chat-message.assistant {
        background-color: #f0f2f5;
        color: black;
        border: 2px solid #ff7e5f;
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

st.markdown('<div class="header">Search Bot</div>', unsafe_allow_html=True)

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

# Add the search input form
with st.form(key='search_form'):
    search_container = st.container()
    with search_container:
        col1, col2 = st.columns([4, 1])
        with col1:
            prompt = st.text_input("Message ChatBOT...", placeholder="Type your message here")
        with col2:
            search_button = st.form_submit_button("Search")

if search_button and prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-message user">{prompt}</div>', unsafe_allow_html=True)

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
