import base64
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import requests
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException


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


class StoryRequest(BaseModel):
    landmark: str
    style: str | None = "neutral"    # e.g. funny, scary, romantic
    tone: str | None = "casual"      # e.g. casual, formal
    length: str | None = "medium"    # short, medium, long


class StoryResponse(BaseModel):
    landmark: str
    summary: str
    story: str



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



def get_wikipedia_summary(landmark: str) -> str | None:
    """
    Returns a short summary for the landmark from Wikipedia.
    If not found, returns None.
    """
    # Replace spaces with underscores for Wikipedia URL
    title = landmark.replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None

        data = resp.json()
        # 'extract' is the short summary field
        return data.get("extract")
    except Exception:
        return None


@app.post("/generate-story", response_model=StoryResponse)
async def generate_story(payload: StoryRequest):
    """
    1. Takes a landmark + user preferences (style, tone, length).
    2. Fetches Wikipedia summary.
    3. Calls OpenAI LLM to generate a customized story.
    4. Returns summary + story.
    """

    wiki_summary = get_wikipedia_summary(payload.landmark)

    if not wiki_summary:
        raise HTTPException(
            status_code=404,
            detail=f"Wikipedia summary not found for landmark '{payload.landmark}'",
        )

    prompt = (
        "You are a storytelling assistant for a mobile tour guide app.\n\n"
        f"Landmark name: {payload.landmark}\n"
        f"User style preference: {payload.style}\n"
        f"User tone preference: {payload.tone}\n"
        f"Requested length: {payload.length}\n\n"
        "Use the following factual information from Wikipedia as your knowledge base:\n"
        "-----\n"
        f"{wiki_summary}\n"
        "-----\n\n"
        "Now write a narrative-style story about this landmark. "
        "Requirements:\n"
        "- Respect the style and tone preferences.\n"
        "- Avoid bullet points; write as a continuous story.\n"
        "- Do NOT mention Wikipedia or that this text came from an API.\n"
        "- Keep it readable on a phone (2–5 short paragraphs).\n"
    )

    # Using the same OpenAI client you already configured
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    )

    story_text = response.output_text.strip()

    return StoryResponse(
        landmark=payload.landmark,
        summary=wiki_summary,
        story=story_text,
    )
