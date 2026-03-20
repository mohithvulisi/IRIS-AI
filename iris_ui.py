import streamlit as st
from langchain_groq import ChatGroq
from ddgs import DDGS
from pypdf import PdfReader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="IRIS", layout="centered")
st.title("IRIS - AI Assistant")

llm = ChatGroq(model="llama-3.3-70b-versatile")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are IRIS, a helpful AI assistant. When you need to search the web, say SEARCH: followed by your query.")]
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file:
        reader = PdfReader(uploaded_file)
        st.session_state.pdf_text = ""
        for page in reader.pages:
            st.session_state.pdf_text += page.extract_text()
        st.success(f"Loaded — {len(reader.pages)} pages")

def search_web(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        return "\n".join([r['body'] for r in results])

def chat(user_input):
    if st.session_state.pdf_text:
        full_input = f"PDF Content:\n{st.session_state.pdf_text[:3000]}\n\nUser Question: {user_input}"
    else:
        full_input = user_input

    st.session_state.chat_history.append(HumanMessage(content=full_input))
    response = llm.invoke(st.session_state.chat_history)

    if "SEARCH:" in response.content:
        query = response.content.split("SEARCH:")[1].strip()
        results = search_web(query)
        st.session_state.chat_history.append(AIMessage(content=response.content))
        st.session_state.chat_history.append(HumanMessage(content=f"Search results: {results}"))
        final = llm.invoke(st.session_state.chat_history)
        st.session_state.chat_history.append(AIMessage(content=final.content))
        return final.content

    st.session_state.chat_history.append(AIMessage(content=response.content))
    return response.content

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask IRIS anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})