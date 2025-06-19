from fastapi import FastAPI, Request
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Tokenizer API is ready."}

@app.post("/tokenize/")
async def tokenize(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return {"input_ids": [], "attention_mask": []}

    inputs = tokenizer(prompt, return_tensors="pt")
    return {
        "input_ids": inputs["input_ids"].tolist(),
        "attention_mask": inputs["attention_mask"].tolist()
    }
