from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from ddgs import DDGS
from pypdf import PdfReader
import io
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
llm = ChatGroq(model="llama-3.3-70b-versatile")
chat_history = [SystemMessage(content="You are IRIS, a highly intelligent AI assistant. When you need to search the web, say SEARCH: followed by your query. Be concise and helpful.")]
pdf_text = ""

class Message(BaseModel):
    text: str

def search_web(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        return "\n".join([r['body'] for r in results])

@app.post("/chat")
async def chat(msg: Message):
    global pdf_text
    if pdf_text:
        full_input = f"PDF Content:\n{pdf_text[:3000]}\n\nUser Question: {msg.text}"
    else:
        full_input = msg.text

    chat_history.append(HumanMessage(content=full_input))
    response = llm.invoke(chat_history)

    if "SEARCH:" in response.content:
        query = response.content.split("SEARCH:")[1].strip()
        results = search_web(query)
        chat_history.append(AIMessage(content=response.content))
        chat_history.append(HumanMessage(content=f"Search results: {results}"))
        final = llm.invoke(chat_history)
        chat_history.append(AIMessage(content=final.content))
        return {"reply": final.content}

    chat_history.append(AIMessage(content=response.content))
    return {"reply": response.content}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global pdf_text
    contents = await file.read()
    reader = PdfReader(io.BytesIO(contents))
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
    return {"pages": len(reader.pages)}

@app.get("/")
async def root():
    return FileResponse("index.html")