
FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3️⃣ System dependencies (OpenCV needs these)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Working directory
WORKDIR /app

# 5️⃣ Install Python dependencies FIRST (cache-friendly)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# 6️⃣ Copy project files
COPY . .

# 7️⃣ Expose FastAPI port
EXPOSE 8001

# 8️⃣ Start app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
