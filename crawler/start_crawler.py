import sys, asyncio

def main():
    # 1) prima riga: patch al loop su Windows
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(
            asyncio.WindowsProactorEventLoopPolicy()
        )

    # 2) poi importiamo uvicorn e la nostra app
    import uvicorn
    from crawler.main import app

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )

if __name__ == "__main__":
    main()