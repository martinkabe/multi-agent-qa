
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI')
llm = ChatOpenAI(model="gpt-4", temperature=0)
