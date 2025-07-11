from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from tools import init_vectorstore, interpreter_tool, retriever_tool, writer_tool


app = FastAPI()
initialized = False  # flag instead of agent


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
    
    # Manual tool chain execution
    refined = interpreter_tool.invoke(question)
    evidence = retriever_tool.invoke(refined)
    final_answer = writer_tool.invoke(evidence)

    return {"answer": final_answer}
