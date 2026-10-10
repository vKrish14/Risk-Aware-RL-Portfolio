FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORTFOLIO_API_URL=http://127.0.0.1:8000 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-space.txt /app/requirements-space.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<3" \
    && python -m pip install --no-cache-dir -r requirements-space.txt

ENV PYTHONPATH=/app

COPY . /app

EXPOSE 7860

CMD ["bash", "scripts/start_space.sh"]
