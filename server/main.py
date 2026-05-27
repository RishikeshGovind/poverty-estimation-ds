"""
AfricaLens API server.

Start with:
    uvicorn server.main:app --reload --port 8000
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.routes import worldbank, acled, ai_query

app = FastAPI(title="AfricaLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(worldbank.router, prefix="/api")
app.include_router(acled.router,     prefix="/api")
app.include_router(ai_query.router,  prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve built React app in production — must come LAST (catch-all)
_dist = os.path.join(os.path.dirname(__file__), "..", "client", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="static")
