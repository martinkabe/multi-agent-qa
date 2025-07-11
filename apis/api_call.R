library(httr2)
library(jsonlite)
 
# Function for the API call
call_api <- function(prompt,system) {
  # Configuration
  #API_URL <- "http://127.0.0.1:8001/generate"  # URL of your FastAPI endpoint
  API_URL <- "http://awsacgnval0031.jnj.com/generate/"
  # Generation parameters for the API call
  GENERATION_PARAMS <- list(
    max_new_tokens = 1024,
    temperature = 0.5,
    top_p = 0.9
  )
  prompt <- paste0("R code: ", prompt)
  api_payload <- c(list(prompt = prompt, system = system), GENERATION_PARAMS)

  req <- request(API_URL) |>
    req_headers("Content-Type" = "application/json") |>
    req_body_json(api_payload) |>
    req_timeout(120)

  resp <- req_perform(req)

  if (resp_status(resp) == 200) {
    api_response <- resp_body_json(resp)
    return(api_response$response)
  } else {
    stop(paste("API Error:", resp_status(resp), "-", resp_body_string(resp)))
  }
}