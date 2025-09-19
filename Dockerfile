FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME="/tmp" \
    GRADIO_TEMP_DIR="/tmp"

WORKDIR /app

# Копіюємо requirements окремо для кешування
COPY requirements.txt .

# Оновлення безпеки + тимчасові build-інструменти (видалимо після збірки)
# Залишаємо тільки runtime-пакети: libmagic1, curl
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
       gcc g++ make \
       libmagic1 curl \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y gcc g++ make \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Копіюємо застосунок
COPY . .

RUN mkdir -p /tmp && chmod 777 /tmp

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

RUN chmod +x /app/app.py

EXPOSE 7860

CMD ["python", "app.py"]