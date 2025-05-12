#!/usr/bin/env python3
import os
import time
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyUrl
from starlette.responses import StreamingResponse

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

# libreria per HTML → Markdown
from markdownify import markdownify as md
# libreria per manipolare l'HTML
from bs4 import BeautifulSoup

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
        # prendi cleaned_html (senza script/stili) o html grezzo
        raw_html = res.cleaned_html or res.html

        # ---- RIMUOVI TAG IMG, SVG, etc. ----
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup.find_all(["img", "svg", "picture", "figure"]):
            tag.decompose()
        cleaned_html = str(soup)
        # ------------------------------------

        # converti l'HTML “depurato” in Markdown
        cleaned_md = md(cleaned_html, heading_style="ATX").strip()

        block = (
            f"## {res.url}\n\n"
            f"{cleaned_md}\n\n"
        )
        yield block.encode()

    duration = time.perf_counter() - start
    yield f"**Pagine crawlate**: {len(results)} • **Tempo**: {duration:.2f}s\n".encode()


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
