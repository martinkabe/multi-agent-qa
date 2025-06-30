from datetime import datetime
import os

from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from config import llm

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = None
retriever_chain = None

def init_vectorstore(document: str):
    global vectorstore, retriever_chain
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([document])
    vectorstore = FAISS.from_documents(docs, embedding_model)
    retriever = vectorstore.as_retriever()
    retriever_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@tool
def interpreter_tool(question: str) -> str:
    """Analyzes and rephrases the user question before retrieval."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert question analyst. Clarify and extract core meaning."),
        ("human", "{question}")
    ])
    return (prompt | llm).invoke({"question": question}).content

@tool
def retriever_tool(query: str) -> str:
    """Searches the document for relevant content using vector similarity."""
    return retriever_chain.invoke({"query": query})["result"]

@tool
def writer_tool(content: str) -> str:
    """Writes a professional answer based on the extracted content and saves it to a markdown file."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a detailed and clear scientific answer."),
        ("human", "Content:\n{content}")
    ])
    
    # Generate the answer
    result = (prompt | llm).invoke({"content": content}).content

    # Save to Markdown file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/answer_{timestamp}.md"
    filepath = os.path.join(".", filename)  # Saves in current working directory

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# AI-Generated Answer\n\n")
        f.write(result)

    return result
