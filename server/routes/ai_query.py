import os
import json
from google import genai
from google.genai import types
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    context: list[dict]   # [{country, iso3, poverty_rate, hdi}, ...]


SYSTEM_PROMPT = """You are AfricaLens, a satellite intelligence analyst specialising in
poverty and human development across Sub-Saharan Africa. Answer concisely (2–4 sentences max)
using the country data provided. Always name specific countries with figures.

If the question is about a specific country, end your response with this tag on its own line:
<fly_to>ISO3_CODE</fly_to>  (e.g. <fly_to>NGA</fly_to>). Omit it for general questions."""


@router.post("/ai-query")
async def ai_query(req: QueryRequest):
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {"answer": "Set GEMINI_API_KEY to enable AI queries.", "fly_to": None}

    client = genai.Client(api_key=api_key)
    data_summary = json.dumps(req.context[:30], indent=2)
    prompt = f"Country data:\n{data_summary}\n\nQuestion: {req.query}"

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=300,
            ),
        )
        raw = response.text.strip()
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            return {"answer": "Gemini free-tier quota reached — try again in a minute.", "fly_to": None}
        return {"answer": f"AI query failed: {err[:120]}", "fly_to": None}

    # Parse optional fly_to tag
    fly_to = None
    if "<fly_to>" in raw and "</fly_to>" in raw:
        start = raw.index("<fly_to>") + 8
        end = raw.index("</fly_to>")
        fly_to = raw[start:end].strip()
        raw = raw[:raw.index("<fly_to>")].strip()

    return {"answer": raw, "fly_to": fly_to}
