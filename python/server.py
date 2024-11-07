from fastapi import FastAPI, Request
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import ORJSONResponse
from mlc_llm import MLCEngine

model: str = None
engines: MLCEngine = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = "HF://HuggingFaceTB/smollm-135M-instruct-add-basics-q0f16-MLC"
    engines["main"] = MLCEngine(model, mode="local")
    yield
    engines["main"].terminate()
    
app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Server is running."}

@app.post("/generate", response_class=ORJSONResponse)
async def generate(req: Request):
    body = await req.json()
    query = body['query']
    
    response = engines["main"].chat.completions.create(
        messages=[{"role": "system", "content": """
                    You will be provided with data based on topics like world hunger, etc. although that may not be specified.\n
                    Past, present and predicted future data will be given as well as the algorithm used to predict it.\n
                    Your job is to create a 1-2 sentence, jargon-free human-readable summary, as people who are not knowledgable in this topic.\n
                    may also wish to see what the data is about, so you need to create a hypothesis/summary to immediately and concisely tell them what is going to happen.\n\n
                    Examples (Take x as past/present data, y as future, predicted data and z as the algorithm used):\n\n
                    Schema for X: {{...},{...},{...}}\n
                    Schema for Y: {{...},{...},{...}}\n
                    Schema for Z: "{STRING}"\n\n
                    
                    First Example:\n\n
                    User: {x}\n\n{y}\n\nAlgorithm Used: {z} (Take country as Afghanistan, data topic as Average Caloric Intake, x as data ranging from 2008-2023, and y as future predicted data ranging from 2024 to any year in the future
                    and z as Holt-Winters Exponential Smoothing)
                    
                    Assistant: Afghanistan's average caloric intake has been on a steady rise, ~1.8% per year. Future data indicate that percentage may increase exponentially as
                    caloric intake increases per year, as predicted by the Holt-Winters Exponential Smoothing Algorithm.
                   """}, {"role": "user", "content": query}],
        model=model,
        stream=False
    )
    
    return ORJSONResponse({"query": query, "response": response.choices[0].message.content})