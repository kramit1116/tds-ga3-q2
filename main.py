import os
import base64
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- Setup ----
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()

# ---- Enable CORS (Rule #2) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow requests from anywhere (the grader's Cloudflare Worker)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Define the shape of the incoming request ----
class ImageQuestion(BaseModel):
    image_base64: str
    question: str

# ---- The actual endpoint ----
@app.post("/answer-image")
def answer_image(payload: ImageQuestion):
    image_bytes = base64.b64decode(payload.image_base64)

    prompt = (
        f"{payload.question}\n\n"
        "Answer with ONLY the raw value — no units, no currency symbols, "
        "no extra words. If it's a number, just give the number."
    )

    response = model.generate_content(
        [
            {"mime_type": "image/png", "data": image_bytes},
            prompt,
        ]
    )

    answer_text = response.text.strip()

    # Rule #1: answer must ALWAYS be a string
    return {"answer": str(answer_text)}
