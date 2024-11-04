import onnxruntime_genai
import fastapi
from fastapi import Request
from fastapi.responses import ORJSONResponse
from functools import lru_cache
import os

model_dir = os.path.join(os.getcwd(), "models/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4")

app = fastapi.FastAPI()

@lru_cache(maxsize=256)
def load_model():
    model = onnxruntime_genai.Model(model_dir)
    tokenizer = onnxruntime_genai.Tokenizer(model)
    print("[INFO/MAIN]: Loaded model and tokenizer into LRU cache")
    return model, tokenizer

@lru_cache(maxsize=128)
def invoke(model: any, tokenizer: any, query: str):
    search_options = {}
    search_options['max_length'] = 8192
    search_options['temperature'] = 0.0
    
    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
    prompt = f'{chat_template.format(input=query)}'
    
    input_tokens = tokenizer.encode(prompt)
    params = onnxruntime_genai.GeneratorParams(model)
    params.try_graph_capture_with_max_batch_size(4)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens

    output_tokens = model.generate(params)

    output_text = tokenizer.decode(output_tokens)

    assistant_response = output_text.split(query)[1].strip()

    return assistant_response

[model, tokenizer] = load_model()

@app.get("/")
def root():
    return {"message": "Server is running."}

@app.post("/generate", response_class=ORJSONResponse)
async def generate(req: Request):
    body = await req.json()
    query = body['query']
    
    msg = invoke(model, tokenizer, query)
    return ORJSONResponse({"query": query, "response": msg})