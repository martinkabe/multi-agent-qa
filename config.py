
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

## OpenAI model
# os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI')
# llm = ChatOpenAI(model="gpt-4", temperature=0)

## Local huggingface model
model_path = f"{os.getenv('MODEL_LOCATION')}/Llama-3.2-1B-Instruct"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

# Create a generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id
)

# LangChain-compatible LLM
llm = HuggingFacePipeline(pipeline=llm_pipeline)
