from langchain_groq import ChatGroq
from ddgs import DDGS
from pypdf import PdfReader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")
chat_history = [SystemMessage(content="""You are IRIS, a helpful AI assistant.
When you need to search the web, say SEARCH: followed by your query.
After seeing search results, give a helpful answer.""")]

pdf_text = ""

def search_web(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        return "\n".join([r['body'] for r in results])

def load_pdf(path):
    global pdf_text
    reader = PdfReader(path)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
    print(f"\n✅ PDF loaded! ({len(reader.pages)} pages)\n")

def chat(user_input):
    # If PDF is loaded, include it as context
    if pdf_text:
        full_input = f"PDF Content:\n{pdf_text[:3000]}\n\nUser Question: {user_input}"
    else:
        full_input = user_input

    chat_history.append(HumanMessage(content=full_input))
    response = llm.invoke(chat_history)

    # If IRIS wants to search
    if "SEARCH:" in response.content:
        query = response.content.split("SEARCH:")[1].strip()
        results = search_web(query)
        chat_history.append(AIMessage(content=response.content))
        chat_history.append(HumanMessage(content=f"Search results: {results}"))
        final = llm.invoke(chat_history)
        chat_history.append(AIMessage(content=final.content))
        return final.content

    chat_history.append(AIMessage(content=response.content))
    return response.content

print(" IRIS ready! Type 'quit' to exit.")
print(" Tip: Type 'load pdf' to load a PDF file!\n")

while True:
    user = input("You: ")
    if user.lower() == "quit":
        break
    elif user.lower() == "load pdf":
        path = input("Enter PDF file path: ")
        load_pdf(path)
    else:
        print(f"\nIRIS: {chat(user)}\n")