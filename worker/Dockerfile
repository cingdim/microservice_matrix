FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Run the API on port 8001 inside container
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8001"]
