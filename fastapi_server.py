from fastapi import FastAPI
from transformers import pipeline
import uvicorn

app = FastAPI()
generator = pipeline('text-generation', model='squeeze-ai-lab/TinyAgent-1.1B', device='mps')

@app.post("/generate")
async def generate_text(prompt: str):
    return generator(prompt)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
