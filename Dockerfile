# Dockerfile.main
FROM python:3.12-slim

WORKDIR /app

# Copy source and requirements
COPY main.py .
COPY auto_test.py .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/results

# Wait for dependent services before starting main.py
CMD ["sh", "-c", "sleep 5 && python main.py"]
