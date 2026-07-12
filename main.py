import os
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- Setup: point the OpenAI SDK at AI Pipe instead of OpenAI directly ----
client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)

app = FastAPI()

# ---- Enable CORS so the grader's Cloudflare Worker can call this API ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageQuestion(BaseModel):
    image_base64: str
    question: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/answer-image")
def answer_image(payload: ImageQuestion):
    prompt = (
        f"{payload.question}\n\n"
        "Answer with ONLY the raw value — no units, no currency symbols, "
        "no extra words. If it's a number, just give the number."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{payload.image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        answer_text = response.choices[0].message.content.strip()
        return {"answer": str(answer_text)}

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream model error: {str(e)}")
