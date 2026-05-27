import httpx
from fastapi import APIRouter, Query
from server.services import data_cache

router = APIRouter()

WB_BASE = "https://api.worldbank.org/v2"
SSA_REGION = "SSF"


@router.get("/worldbank/poverty")
async def get_poverty(year: int = Query(2023, ge=2000, le=2023)):
    cache_key = f"wb_poverty_{year}"
    cached = data_cache.get(cache_key)
    if cached:
        return cached

    url = (
        f"{WB_BASE}/country/{SSA_REGION}/indicator/SI.POV.DDAY"
        f"?date={year}&format=json&per_page=100"
    )
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        raw = r.json()

    data = raw[1] if isinstance(raw, list) and len(raw) > 1 else []
    result = {
        "year": year,
        "records": [
            {
                "iso3": row["countryiso3code"],
                "country": row["country"]["value"],
                "value": row["value"],
            }
            for row in data
            if row.get("value") is not None
        ],
    }
    data_cache.set(cache_key, result)
    return result
