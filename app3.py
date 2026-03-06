import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

##ARXIV tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250,
                              arxiv_search=any,arxiv_exceptions=any)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

###wikipedia Tools
wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250,wiki_client=any)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)


search = DuckDuckGoSearchResults(name="Web Search")


load_dotenv()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔎 GEN AI Search Engine using Agents & Tools")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API Key:", type="password")

# -----------------------------
# Initialize Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi 👋 I can search Arxiv, Wikipedia and the Web. Ask me anything!"
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------------
# User Input
# -----------------------------
if prompt := st.chat_input("Ask me anything..."):

    if not api_key:
        st.warning("Please enter your GROQ API key in sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # -----------------------------
    # Initialize LLM (Active Groq Model)
    # -----------------------------
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",  # Active Groq model
        streaming=True
    )

    # -----------------------------
    # Tools Setup

    tools = [search, arxiv, wiki]

    # -----------------------------
    # Agent Initialization
    # -----------------------------
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # -----------------------------
    # Generate Response
    # -----------------------------
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(prompt, callbacks=[st_cb])
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})