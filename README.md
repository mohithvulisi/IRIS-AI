# IRIS-AI

An intelligent conversational AI assistant built with LangChain, Groq (Llama-3.3-70B), DuckDuckGo web search, and PDF document context analysis.

## Features
- **Conversational AI**: Powered by Llama-3.3-70B via Groq for high-speed inference.
- **Web Search**: Real-time information retrieval using DuckDuckGo.
- **Document Q&A**: PDF ingestion and context extraction using PyPDF.
- **Dual Interfaces**: REST API / Modern Web UI (FastAPI) and Streamlit interactive dashboard.

## Setup & Running

1. Clone the repository:
   ```bash
   git clone https://github.com/mohithvulisi/IRIS-AI.git
   cd IRIS-AI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory from the template:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Run the FastAPI server:
   ```bash
   uvicorn api:app --reload
   ```
   Or run the Streamlit UI:
   ```bash
   streamlit run iris_ui.py
   ```
