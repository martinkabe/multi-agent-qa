from datetime import datetime
import os

from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from config import llm

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=f"{os.getenv('MODEL_LOCATION')}/all-MiniLM-L6-v2"
)

# Global store
vectorstore = None
retriever_chain = None

def init_vectorstore(document: str):
    """Splits the document, creates vectorstore, and initializes retriever chain."""
    global vectorstore, retriever_chain
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([document])
    vectorstore = FAISS.from_documents(docs, embedding_model)
    retriever = vectorstore.as_retriever()
    retriever_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@tool
def interpreter_tool(question: str) -> str:
    """Rephrases the user question for clarity and relevance."""
    prompt = PromptTemplate.from_template(
        "You are a scientific assistant. Rephrase the user's question to make it clearer and more specific.\n\n"
        "Original Question: {question}\n\nRefined Question:"
    )
    return (prompt | llm).invoke({"question": question})

@tool
def retriever_tool(query: str) -> str:
    """Queries the vectorstore and returns the most relevant extracted content."""
    return retriever_chain.invoke({"query": query})["result"]

@tool
def writer_tool(context: str, question: str) -> str:
    """Writes a grounded scientific answer using context and saves to a markdown file."""
    prompt = PromptTemplate.from_template(
        "You are a scientific assistant.\n"
        "Answer the following user question based solely on the provided context.\n"
        "If the context does not contain the answer, respond with 'Not available in provided data.'\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )

    result = (prompt | llm).invoke({"context": context, "question": question})

    os.makedirs("reports", exist_ok=True)
    filename = f"reports/answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# AI-Generated Answer\n\n")
        f.write(f"**Question:** {question}\n\n")
        f.write(result.strip())

    return result.strip()
