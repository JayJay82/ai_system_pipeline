#!/usr/bin/env python3
import os
import time
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyUrl
from starlette.responses import StreamingResponse

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

app = FastAPI(title="Crawler API")


class CrawlRequest(BaseModel):
    url: AnyUrl
    depth: Optional[int]   = 3
    timeout: Optional[int] = int(os.getenv("CRWL_TIMEOUT", 30))


async def crawl_generator(
    url: str,
    max_depth: int,
    timeout:   int
) -> AsyncGenerator[bytes, None]:
    yield f"# Risultati di crawl di {url}\n\n".encode()
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=max_depth,
            include_external=False
        ),
        stream=False
    )
    start = time.perf_counter()
    async with AsyncWebCrawler(timeout=timeout) as crawler:
        results = await crawler.arun(url=url, config=run_config)

    for res in results:
        block = (
            f"## {res.url}\n\n"
            f"{(res.markdown if res.success else res.error_message)}\n\n"
        )
        yield block.encode()

    duration = time.perf_counter() - start
    yield f"**Pagine: {len(results)}, Tempo: {duration:.2f}s**".encode()


@app.post("/crawl")
async def crawl_endpoint(request: CrawlRequest):
    try:
        gen = crawl_generator(str(request.url), request.depth, request.timeout)
        fname = f"crawl_{request.url.host}.md"
        return StreamingResponse(
            gen,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'}
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Errore durante il crawl: {e}")
