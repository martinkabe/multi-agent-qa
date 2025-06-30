
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from planner import create_agent
from tools import init_vectorstore

app = FastAPI()
agent = None

@app.post("/initialize/")
async def initialize(file: UploadFile):
    content = await file.read()
    document = content.decode("utf-8")
    init_vectorstore(document)
    global agent
    agent = create_agent()
    return {"status": "initialized"}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    if agent is None:
        return JSONResponse(content={"error": "Agent not initialized"}, status_code=400)
    result = agent.invoke({"input": question})
    return {"answer": result["output"]}

# uvicorn main:app --reload