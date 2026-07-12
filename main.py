import os
import re
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)

app = FastAPI()

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

def clean_answer(text: str) -> str:
    text = text.strip()
    # strip common wrapping characters models add
    text = text.strip('"\'`')
    # remove currency symbols and thousands separators
    text = re.sub(r'[$,₹€£]', '', text)
    text = text.strip()
    return text

@app.post("/answer-image")
def answer_image(payload: ImageQuestion):
    prompt = (
        f"Look carefully at this image and answer the following question:\n"
        f"{payload.question}\n\n"
        "Rules:\n"
        "- Respond with ONLY the raw value, nothing else.\n"
        "- No units, no currency symbols, no commas, no explanations.\n"
        "- If the answer is a number, give it exactly as it appears/computes "
        "(preserve decimal places, do not round unless asked).\n"
        "- Read all text and numbers carefully before answering, especially "
        "small print in tables and invoices."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{payload.image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
            temperature=0,
        )
        raw_answer = response.choices[0].message.content
        answer = clean_answer(raw_answer)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream model error: {str(e)}")
