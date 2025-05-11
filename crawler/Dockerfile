# Dockerfile multi-stage per Crawl4AI con FastAPI

# Stage di build
FROM python:3.11-slim AS builder
WORKDIR /app

# Installa le librerie di sistema necessarie per Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libatk-bridge2.0-0 libx11-xcb1 libdrm2 libgbm1 libgtk-3-0 \
    libxcomposite1 libxdamage1 libxrandr2 libasound2 ca-certificates fonts-liberation \
    libglib2.0-0 libpng16-16 libjpeg62-turbo && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Installa browser Chromium per Playwright
RUN playwright install --with-deps chromium

# Stage finale
FROM python:3.11-slim
WORKDIR /app

# Installa runtime di sistema per Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libatk-bridge2.0-0 libx11-xcb1 libdrm2 libgbm1 libgtk-3-0 \
    libxcomposite1 libxdamage1 libxrandr2 libasound2 ca-certificates fonts-liberation \
    libglib2.0-0 libpng16-16 libjpeg62-turbo && rm -rf /var/lib/apt/lists/*

# Copia pacchetti Python e binari da builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia i browser installati da Playwright
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

# Copia l'app e i requirements
COPY main.py ./
COPY requirements.txt ./

# Espone la porta dell'API
EXPOSE 8000

# Avvia il server Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]