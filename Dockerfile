FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ .
COPY data/ ./data/

EXPOSE 8501

CMD ["streamlit", "run", "frontend.py", "--server.address=0.0.0.0", "--server.port=8501"]