Download LLMs via Huggingface Github

```bash
# for vector embeddings
git clone git@hf.co:sentence-transformers/all-mpnet-base-v2
# LLM instruct
git clone git@hf.co:meta-llama/Llama-3.2-1B-Instruct
```

Run API via uvicorn

```bash
uvicorn main:app --reload
```

Run Streamlit app

```bash
streamlit run app_planner.py
```

Q for AI:

Can you explain how the storage temperature affects API-X stability and what it means for shelf life?
What are the primary degradation products?


⚡ What is Uvicorn?

**Uvicorn** is a lightning-fast **ASGI** web server implementation for Python.

* ASGI stands for **Asynchronous Server Gateway Interface**, and it's the modern standard for Python web apps that want to support WebSockets, long-polling, async APIs, etc.

Uvicorn is built on:

* **uvloop**: an ultra-fast event loop built on top of libuv (used by Node.js)

* **httptools**: a fast HTTP parser

🧩 Why Uvicorn Is Used With FastAPI

**FastAPI** is an **ASGI** web framework — not a web server itself.
That means you need something to serve HTTP requests and route them to your FastAPI app. That's where Uvicorn comes in.


✅ Uvicorn Responsibilities:

* Starts a web server

* Listens for HTTP (or WebSocket) connections

* Converts incoming requests into ASGI messages

* Sends those messages to your FastAPI app

* Returns the response to the client


🤗 Hugging Face — What It Is

**Hugging Face** is a company and open-source ecosystem focused on natural language processing (NLP) and machine learning. It provides:

🔧 Tools and Libraries:

* **Transformers**: a massively popular Python library that gives you access to pre-trained models like BERT, GPT-2, T5, etc.

* **Datasets**: a collection of ready-to-use NLP datasets

* **Tokenizers**: efficient tools for converting text to model-ready format


🔗 LangChain — What It Is

**LangChain** is a Python framework for building applications with Large Language Models (LLMs). It helps developers connect LLMs like GPT-4 to tools, memory, databases, and logic.

🧩 Key Features:

* **Chains**: build multi-step workflows (e.g. question → interpret → retrieve → answer)

* **Agents**: let the LLM plan its own steps using tools

* **Memory**: retain context across user interactions

* **Tools Integration**: plug-in calculators, search engines, retrievers, databases


## 🎯 Main Purpose of This App

The goal of this application is to demonstrate how LLM-based agents can work together to extract, reason, and write human-quality answers from long or technical documents — a common need in regulated industries such as pharmaceuticals, legal, or compliance-driven fields.

## 🧾 What This App Does

This application is a multi-agent question answering system built with FastAPI and LangChain. It allows users to upload a domain-specific document and ask natural language questions. Behind the scenes, a modular chain of agents (interpreter, retriever, writer) collaborates to generate accurate, context-aware answers.

* Each agent plays a distinct role:

* The interpreter agent reformulates complex or ambiguous questions

* The retriever agent uses semantic search (via FAISS + sentence transformers) to find relevant context

* The writer agent produces a professional answer and saves it as a Markdown report


📄 What’s in document.txt

The sample document.txt file contains a pharmaceutical stability report describing the degradation behavior of API-X (a fictional Active Pharmaceutical Ingredient) under different storage conditions. It includes:

* Environmental testing methodology

* Degradation product data

* Shelf-life findings

* Scientific conclusions

## Application Components

🔧 config.py — LLM Configuration

🔍 What it does:

* Stores and initializes the connection to OpenAI's GPT-4.

* Sets `temperature=0` for deterministic responses.

* `llm` is imported and reused across all components.


🧠 planner.py — Agent Initialization

🔍 What it does:

* Imports 4 modular tools (LLM-based functions).

* Wraps them into a LangChain OpenAI Functions Agent.

* `verbose=True` logs tool calls (great for trace debugging).

When you call `agent.invoke({"input": question})`, the LLM:

* Parses the question

* Picks and runs tools step by step

* Returns the final result

🛠️ tools.py — LLM Tools for Agents

🔧 Tool 1: interpreter_tool

🔧 Tool 2: retriever_tool

🔧 Tool 3: writer_tool


🧬 How It All Works Together
1. User uploads a document:

* init_vectorstore() splits it and builds FAISS index.

This function is critical infrastructure for your multi-agent system. It prepares the document for semantic search, which enables the retriever agent to find the most relevant parts of the document when answering a user question.

✅ 1. Purpose of the Vector Store

When a user uploads a document (like document.txt), it's too long for GPT to read all at once. Instead, we:

* Split the document into small chunks

* Convert each chunk into a **vector** (a list of numbers representing meaning)

* Store those vectors in a **FAISS index** (a searchable database optimized for similarity search)

This enables **semantic retrieval**: finding the most relevant chunk of text based on meaning, not just keyword match.


🔍 What the Function Does — Step-by-Step

```python
def init_vectorstore(document: str):
    global vectorstore, retriever_chain
```

* Initializes two global variables: vectorstore and retriever_chain so other parts of the app can use them.

🔹 Step 1: Chunk the document

```python
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([document])
```

* Splits the long document into overlapping chunks of 500 characters (with 50-character overlap).

* This overlap prevents losing important context between boundaries.

*Example:*

```perl
Chunk 1: "API-X is tested at 25°C..."
Chunk 2: "...25°C/60%RH for 12 months..."
```

🔹 Step 2: Embed the chunks into vectors

```python
vectorstore = FAISS.from_documents(docs, embedding_model)
```

* Each chunk is transformed into a high-dimensional vector using a sentence-transformer embedding model.

* These vectors are stored in a **FAISS index** — optimized for fast similarity search.

🔹 Step 3: Build the retrieval chain


```python
retriever = vectorstore.as_retriever()
retriever_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

* Converts the FAISS vector store into a LangChain-compatible retriever

* Wraps that in a RetrievalQA chain so it can automatically:

   * Accept a question

   * Retrieve relevant chunks

   * Pass them to the LLM to answer


2. User asks a question:

* agent.invoke(...) runs the OpenAI LLM planner.

3. Planner decides which tools to invoke:

* Usually: interpreter_tool → retriever_tool → writer_tool.

4. Output is displayed and saved:

* Writer saves .md file to reports/.