FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e ".[dev]"

FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

EXPOSE 8501

CMD ["streamlit", "run", "vectorbenchapp/Get_Started.py", "--server.address=0.0.0.0", "--server.port=8501"]
