import os
import base64
from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- Setup ----
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = FastAPI()

# ---- Enable CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageQuestion(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
def answer_image(payload: ImageQuestion):
    prompt = (
        f"{payload.question}\n\n"
        "Answer with ONLY the raw value — no units, no currency symbols, "
        "no extra words. If it's a number, just give the number."
    )

    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-5" if you have access
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
