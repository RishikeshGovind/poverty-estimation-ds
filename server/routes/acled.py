import os
import httpx
from fastapi import APIRouter, Query
from server.services import data_cache

router = APIRouter()

ACLED_BASE = "https://api.acleddata.com/acled/read"


@router.get("/acled")
async def get_acled(year: int = Query(2023, ge=2000, le=2024)):
    cache_key = f"acled_{year}"
    cached = data_cache.get(cache_key)
    if cached:
        return cached

    api_key = os.getenv("ACLED_API_KEY", "")
    email = os.getenv("ACLED_EMAIL", "")

    if not api_key or not email:
        return {"events": [], "note": "Set ACLED_API_KEY and ACLED_EMAIL env vars"}

    params = {
        "key": api_key,
        "email": email,
        "year": year,
        "region": "1|2|3|4|5",   # African regions
        "limit": 200,
        "fields": "event_id_cnty|event_date|event_type|country|latitude|longitude|fatalities|notes",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(ACLED_BASE, params=params)
        r.raise_for_status()
        raw = r.json()

    events = [
        {
            "id":         row["event_id_cnty"],
            "date":       row["event_date"],
            "event_type": row["event_type"],
            "country":    row["country"],
            "lat":        float(row["latitude"]),
            "lon":        float(row["longitude"]),
            "fatalities": int(row.get("fatalities", 0)),
            "notes":      row.get("notes", ""),
        }
        for row in raw.get("data", [])
    ]
    result = {"year": year, "events": events}
    data_cache.set(cache_key, result)
    return result
