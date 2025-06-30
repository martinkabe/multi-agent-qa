
from config import llm
from langchain.agents import initialize_agent, AgentType
from tools import interpreter_tool, retriever_tool, writer_tool

def create_agent():
    tools = [interpreter_tool, retriever_tool, writer_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    return agent
