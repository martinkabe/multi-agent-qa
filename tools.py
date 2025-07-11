from datetime import datetime
import os

from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from config import llm

# Use environment variable to locate model locally
embedding_model = HuggingFaceEmbeddings(model_name=f"{os.getenv('MODEL_LOCATION')}/all-mpnet-base-v2")

# Global objects
vectorstore = None
retriever_chain = None

def init_vectorstore(document: str):
    """Splits the document, creates vectorstore, and initializes the retriever chain."""
    global vectorstore, retriever_chain
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([document])
    vectorstore = FAISS.from_documents(docs, embedding_model)
    retriever = vectorstore.as_retriever()
    retriever_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@tool
def interpreter_tool(question: str) -> str:
    """Rephrases the user question to improve retrieval accuracy."""
    prompt = PromptTemplate.from_template(
        "Clarify and simplify this scientific question:\n\n{question}\n\nRephrased:"
    )
    return (prompt | llm).invoke({"question": question})

@tool
def retriever_tool(query: str) -> str:
    """Queries the vectorstore and returns the most relevant extracted content."""
    return retriever_chain.invoke({"query": query})["result"]

@tool
def writer_tool(content: str) -> str:
    """Writes a professional scientific answer and saves it to a markdown file."""
    
    # Clean instruct-style prompt (no chat roles, no trailing 'Answer:')
    prompt = PromptTemplate.from_template(
        "You are a scientific writing assistant.\n\nWrite a clear and professional scientific answer based on the following content:\n\n{content}"
    )

    # Generate the answer
    result = (prompt | llm).invoke({"content": content})

    # Optional post-processing: extract last sentence or remove duplication
    if isinstance(result, str) and "Answer:" in result:
        result = result.split("Answer:")[-1].strip()

    # Save to Markdown file
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/answer_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# AI-Generated Answer\n\n")
        f.write(result)

    return result
