FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY VERSION README.md ./
COPY configs ./configs
COPY models/calibrated_model.pkl models/preprocessor.pkl models/feature_columns.json ./models/
COPY scripts ./scripts
COPY src ./src
COPY templates ./templates

RUN python -m compileall src scripts

CMD ["python", "scripts/analyze_file.py", "--help"]