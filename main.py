import base64
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

app = FastAPI(
    title="TourGuideAI Vision Backend",
    description="Receives image - sends json with location name",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LandmarkResponse(BaseModel):
    landmark_name: str
    raw_model_response: str


@app.post("/recognize-landmark", response_model=LandmarkResponse)
async def recognize_landmark(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await image.read()

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{image.content_type};base64,{b64_image}"

        prompt = (
            "You are a landmark recognition service. "
            "Look at this image and identify the SINGLE most likely famous landmark, "
            "monument, building, or well-known place of interest. "
            "If you cannot confidently identify a specific landmark, respond exactly with: Unknown landmark.\n\n"
            "Respond with ONLY the name of the landmark or 'Unknown landmark'."
        )

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )
        model_text = response.output_text.strip()
        if model_text.lower().strip() in {"unknown", "unknown landmark", "not a landmark"}:
            landmark_name = "Unknown landmark"
        else:
            landmark_name = model_text.strip().strip('"').strip("'")

        return LandmarkResponse(
            landmark_name=landmark_name,
            raw_model_response=model_text,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision processing failed: {e}")

#docker-compose up --build -d