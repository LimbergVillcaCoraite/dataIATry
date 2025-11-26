FROM python:3.12-slim AS base

WORKDIR /app

# Install build deps, install python packages, then remove build deps to keep image small
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy sources
COPY . /app

# Remove build deps to reduce final image size
RUN apt-get purge -y --auto-remove build-essential \
 && rm -rf /var/lib/apt/lists/* /root/.cache/pip

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]