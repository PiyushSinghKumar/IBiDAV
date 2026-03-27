from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ibidav.service import CATEGORY_PAGE_SIZE, DEFAULT_PAGE_SIZE, LOAD_MORE_PAGE_SIZE, service


BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.warmup_async()
    yield


app = FastAPI(
    title="IBiDAV",
    summary="FastAPI interface for biomedical article exploration and topic discovery.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={
            "request": request,
            "app_name": "IBiDAV",
            "default_page_size": DEFAULT_PAGE_SIZE,
            "load_more_page_size": LOAD_MORE_PAGE_SIZE,
            "category_page_size": CATEGORY_PAGE_SIZE,
        },
    )


@app.get("/api/summary")
async def get_summary() -> dict:
    return service.summary()


@app.get("/api/health")
async def healthcheck() -> dict:
    return {"status": "ok", "artifacts_present": service.summary()["artifacts_present"]}


@app.get("/api/search")
async def search_articles(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    init_state = service.initialization_state()
    return {
        "query": q,
        "limit": limit,
        "results": service.search(q, limit=limit),
        "warming_up": not init_state["ready"],
        "init_state": init_state,
    }


@app.get("/api/categories")
async def list_categories() -> dict:
    return {"categories": service.category_counts()}


@app.get("/api/categories/{category_name}")
async def get_category(
    category_name: str,
    q: str = "",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=CATEGORY_PAGE_SIZE, ge=1, le=24),
) -> dict:
    category_names = service.category_counts().keys()
    if category_name not in category_names:
        raise HTTPException(status_code=404, detail=f"Unknown category: {category_name}")
    return service.category_results(category_name, query=q, offset=offset, limit=limit)
