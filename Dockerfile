# Stage 1: Build the React frontend statically
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Set up the Python FastAPI backend
FROM python:3.11-slim
WORKDIR /app

# Install system compilation packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source code
COPY backend/ ./backend

# Copy compiled frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose the API and unified server port
EXPOSE 8000

# Set production environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Start production ASGI server
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
