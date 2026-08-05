# JetPilotGuard service image.
#
# Teaching note: we use a slim Python base to keep the image small, install the
# package, train the model at build time so the image ships ready-to-serve, and
# launch the FastAPI app with uvicorn. Multi-stage builds could shrink this
# further, but clarity wins for a portfolio project.

FROM python:3.11-slim

# Avoid interactive prompts and .pyc clutter; unbuffered logs for docker logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching: deps change less than code).
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY scripts ./scripts

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[app,mcp,explain]"

# Train the model at build time so the container is ready to serve on start.
RUN python -m scripts.train_model

EXPOSE 8000

# uvicorn serves the FastAPI app; 0.0.0.0 so it's reachable from outside the
# container.
CMD ["uvicorn", "jetpilotguard.io.service:app", "--host", "0.0.0.0", "--port", "8000"]
