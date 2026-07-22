FROM python:3.10-slim

WORKDIR /app

# Install CPU-only torch first so sentence-transformers' dependency resolution
# picks up this (much smaller) build instead of pulling the default CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# requirements.txt lives at app/requirements.txt in this repo, not the root.
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project so the `app` package layout (app/main.py importing
# `app.graph`) is preserved inside the image at /app/app/...
COPY . .

EXPOSE 8000

# Render (and most PaaS providers) inject $PORT at runtime and expect the
# service to bind to it; default to 8000 for local/other environments.
# Must run as app.main:app, not main:app, since main.py does `from app.graph
# import ...` and expects to be imported as part of the `app` package.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
