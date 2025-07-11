import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field # Use Field for examples in Swagger UI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from threading import Lock # For potential thread safety if needed
import logging
import gc # Garbage collector

# --- Configuration ---
# Switch back to Llama 3 Instruct model.
# Requires access approval on Hugging Face for the logged-in user.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- Performance & Memory Tuning ---
# Set USE_QUANTIZATION to True to load the model in 4-bit.
# Requires `bitsandbytes` (works best on Linux/Windows with compatible GPUs).
USE_QUANTIZATION = False # Keep False unless specifically needed

# Set DEVICE_MAP to "auto" to automatically use GPU if available, otherwise CPU.
DEVICE_MAP = "auto"

# Determine the optimal torch dtype based on hardware support.
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
llm_pipeline = None
pipeline_lock = Lock()

# --- FastAPI Application ---
app = FastAPI(
    title="Local Llama 3 API",
    description="API to interact with a locally hosted Meta Llama 3 Instruct model using Hugging Face Transformers.",
    version="1.3.0", # Incremented version
)

# --- Pydantic Models for API Data Structure ---
class PromptRequest(BaseModel):
    """Defines the expected input data for the /generate endpoint."""
    prompt: str = Field(..., description="The user's query or instruction for the LLM.", example="Explain quantum entanglement in simple terms.")
    system: str = Field(..., description="The system instruction for the LLM.", example="Explain quantum entanglement in simple terms.")
    max_new_tokens: int = Field(256, gt=0, le=4096, description="Maximum number of new tokens to generate.")
    temperature: float = Field(0.6, ge=0.0, le=2.0, description="Controls randomness (0=deterministic, >1 more random).")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter.")

class GenerationResponse(BaseModel):
    """Defines the structure of the response sent back by the /generate endpoint."""
    response: str = Field(..., description="The generated text from the LLM.")
    error: str | None = Field(None, description="Contains an error message if generation failed, otherwise null.")

# --- Model Loading Logic (Executed Once on API Startup) ---
@app.on_event("startup")
async def load_model_on_startup():
    """Loads the tokenizer and LLM model into memory when the FastAPI server starts."""
    global llm_pipeline, DEVICE_MAP, USE_QUANTIZATION
    logger.info(f"API startup sequence initiated. Loading model: {MODEL_ID}")

    if not torch.cuda.is_available():
        logger.warning("CUDA (GPU) not available. Model will load on CPU, which will be VERY slow.")
        DEVICE_MAP = "cpu"

    try:
        quantization_config = None
        if USE_QUANTIZATION:
            if DEVICE_MAP == "cpu":
                 logger.warning("Quantization (bitsandbytes) disabled on CPU.")
                 USE_QUANTIZATION = False
            else:
                logger.info("Attempting to enable 4-bit quantization using bitsandbytes.")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=TORCH_DTYPE,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info(f"Quantization config: {quantization_config}")

        logger.info(f"Loading tokenizer: {MODEL_ID}")
        # Llama 3 doesn't typically require trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        logger.info("Tokenizer loaded.")

        logger.info(f"Loading model: {MODEL_ID} with dtype: {TORCH_DTYPE}, device_map: {DEVICE_MAP}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            device_map=DEVICE_MAP,
            quantization_config=quantization_config if USE_QUANTIZATION else None,
            # trust_remote_code=False # Default, Llama 3 usually doesn't need this
            # attn_implementation="flash_attention_2" # Optional: Use Flash Attention 2 if installed/supported
        )
        logger.info("Model loaded successfully.")

        logger.info("Creating text-generation pipeline...")
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        logger.info("Text-generation pipeline created successfully. API is ready.")

    except ImportError as e:
         logger.error(f"ImportError: {e}. Required library might be missing (e.g., bitsandbytes for quantization).")
         llm_pipeline = None
    except Exception as e:
        # Check specifically for huggingface_hub.utils.GatedRepoError or similar if access denied
        if "GatedRepoError" in str(type(e)) or "401 Client Error" in str(e) or "403 Client Error" in str(e):
             logger.error(f"Access Denied Error: Failed to download '{MODEL_ID}'. Please ensure you have requested and been granted access to this model on Hugging Face: https://huggingface.co/{MODEL_ID}")
        logger.exception(f"Fatal error during model loading: {e}") # Log full traceback
        llm_pipeline = None

# --- API Endpoints ---
@app.get("/", summary="API Root/Health Check", tags=["General"])
async def root():
    """Provides a basic health check message indicating the API is running."""
    status = "ready" if llm_pipeline else "error (model not loaded)"
    return {
        "message": f"Llama 3 API is running (Model Status: {status}).",
        "model_id": MODEL_ID,
        "quantization_enabled": USE_QUANTIZATION,
        "device_map": DEVICE_MAP,
        "torch_dtype": str(TORCH_DTYPE),
        "docs_url": "/docs"
        }

@app.post("/generate", response_model=GenerationResponse, summary="Generate Text with Llama 3", tags=["LLM"])
async def generate_text_endpoint(request: PromptRequest):
    """
    Receives a prompt and parameters, generates text using the loaded Llama 3 model,
    and returns the assistant's response. Uses the Llama 3 Instruct chat template via apply_chat_template.
    """
    global llm_pipeline
    if llm_pipeline is None:
        logger.error("Attempted to generate text, but LLM Pipeline is not available.")
        raise HTTPException(status_code=503, detail="Service Unavailable: LLM Pipeline failed to load during startup.")

    logger.info(f"Received generation request: max_tokens={request.max_new_tokens}, temp={request.temperature}, top_p={request.top_p}, prompt='{request.prompt[:80]}, system='{request.system[:80]}...'")

    # --- Use apply_chat_template for Llama 3 Instruct ---
    # This function correctly formats the prompt using Llama 3's specific template.
    messages = [
        {"role": "system", "content": request.system},
        {"role": "user", "content": request.prompt},
    ]
    try:
        prompt_formatted = llm_pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True # Crucial for instruct/chat models
        )
        logger.info(f"Applied Llama 3 chat template.")
    except Exception as e:
        logger.exception(f"Error applying chat template: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: Failed to apply chat template. Error: {str(e)}")

    # Define generation parameters
    generation_args = {
        "max_new_tokens": request.max_new_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "do_sample": True if request.temperature > 0 else False,
        # Define End-of-Sequence tokens for Llama 3.
        "eos_token_id": [
            llm_pipeline.tokenizer.eos_token_id,
            llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>") # Llama 3 specific End Of Turn token
            ],
        "pad_token_id": llm_pipeline.tokenizer.eos_token_id
    }

    try:
        logger.info("Invoking LLM pipeline...")
        with pipeline_lock:
            output = llm_pipeline(prompt_formatted, **generation_args)
        logger.info("LLM pipeline execution finished.")

        # --- Response Parsing ---
        full_generated_text = output[0]['generated_text']
        # Find the start of the assistant's response based on the template structure
        assistant_response_start_marker = "<|start_header_id|>assistant<|end_header_id|>"
        marker_index = full_generated_text.rfind(assistant_response_start_marker)

        if marker_index != -1:
            # Extract text after the marker
            assistant_response = full_generated_text[marker_index + len(assistant_response_start_marker):].strip()
            # Remove potential trailing EOS/EOT tokens
            if assistant_response.endswith(llm_pipeline.tokenizer.eos_token):
                assistant_response = assistant_response[:-len(llm_pipeline.tokenizer.eos_token)].strip()
            if assistant_response.endswith("<|eot_id|>"):
                 assistant_response = assistant_response[:-len("<|eot_id|>")].strip()
            logger.info(f"Generated response (parsed): '{assistant_response[:80]}...'")
        else:
            logger.warning("Could not find the assistant marker in the generated text. Returning text after formatted prompt as fallback.")
            assistant_response = full_generated_text[len(prompt_formatted):].strip() # Less reliable fallback

        # Clean up memory
        del output
        del full_generated_text
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return GenerationResponse(response=assistant_response)

    except Exception as e:
        logger.exception(f"Error during text generation pipeline: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: Failed during generation. Error: {str(e)}")

# --- Optional: Cleanup on Shutdown ---
@app.on_event("shutdown")
async def cleanup_on_shutdown():
    """Releases resources when the API server shuts down."""
    global llm_pipeline
    logger.info("API shutdown sequence initiated. Releasing resources...")
    if llm_pipeline:
        try:
             if hasattr(llm_pipeline, 'model') and llm_pipeline.model: del llm_pipeline.model
             if hasattr(llm_pipeline, 'tokenizer') and llm_pipeline.tokenizer: del llm_pipeline.tokenizer
             del llm_pipeline
             llm_pipeline = None
             logger.info("Pipeline object deleted.")
             gc.collect()
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()
                 logger.info("CUDA cache cleared.")
        except Exception as e:
             logger.error(f"Error during resource cleanup: {e}")
    logger.info("Shutdown cleanup finished.")

# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly from script...")
    uvicorn.run("main:app", host="awsacgnval0031.jnj.com", port=8001, reload=False, workers=1)
