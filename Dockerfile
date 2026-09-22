FROM python:3.14.7-slim
LABEL org.opencontainers.image.source="https://github.com/L0ckR/seif-pii"
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_PYTHON_INSTALL_DIR=/opt/python
RUN pip install --no-cache-dir uv==0.12.17 && uv python install 3.14.7t && uv venv /opt/venv --python 3.14.7t
ENV PATH="/opt/venv/bin:$PATH" SEIF_REQUIRE_FREE_THREADING=1
ENV SEIF_HOST=0.0.0.0 SEIF_PORT=8000 SEIF_WORKERS=3
COPY pyproject.toml requirements.lock ./
RUN uv pip install --python /opt/venv/bin/python -r requirements.lock && useradd --uid 10001 --create-home seif
COPY seif ./seif
COPY web ./web
COPY config ./config
COPY scripts/serve.py ./scripts/serve.py
USER seif
EXPOSE 8000
CMD ["python", "scripts/serve.py"]
