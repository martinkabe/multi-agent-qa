from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from tools import init_vectorstore, interpreter_tool, retriever_tool, writer_tool

app = FastAPI()
initialized = False  # Flag to check if vectorstore is initialized

@app.post("/initialize/")
async def initialize(file: UploadFile):
    content = await file.read()
    document = content.decode("utf-8")
    init_vectorstore(document)
    global initialized
    initialized = True
    return {"status": "initialized"}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    if not initialized:
        return JSONResponse(content={"error": "Document not initialized"}, status_code=400)

    # Run toolchain
    refined = interpreter_tool.invoke({"question": question})
    evidence = retriever_tool.invoke({"query": refined})
    final_answer = writer_tool.invoke({"context": evidence, "question": question})

    return {"refined_question": refined, "evidence": evidence, "answer": final_answer}
