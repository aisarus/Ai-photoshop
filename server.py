from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI()

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

class TextRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate_text")
def generate_text(req: TextRequest):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(req.prompt)
    return {"text": response.text}

@app.post("/generate_image")
def generate_image(req: ImageRequest):
    model = genai.GenerativeModel("gemini-2.5-flash-image")
    response = model.generate_content(req.prompt)

    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            return {"image": part.inline_data.data}

    return {"error": "Image not generated"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
