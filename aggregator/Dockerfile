# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the dependencies first (helps cache them)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all remaining files into the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8002

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "aggregator:app", "--host", "0.0.0.0", "--port", "8002"]
