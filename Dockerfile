# ---- Base image ----
FROM python:3.11-slim

# ---- Environment ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- System Dependencies ----
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic-dev \
 && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Install dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy app ----
COPY . .

# ---- Expose port ----
EXPOSE 8000

# ---- Run FastAPI (Uvicorn) ----
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]