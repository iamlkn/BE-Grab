FROM python:3.10.11-buster as builder

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install system dependencies if any (e.g., for psycopg2-binary if not using slim or for other libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev gcc

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# Assuming your 'app' directory (with main.py) and other BE code are at the same level as this Dockerfile
# or directly within the context being copied.
# If BE/ is your context and your structure is BE/app/main.py, then COPY . . is fine.
COPY . .

# Expose the port your application listens on.
# Cloud Run expects the container to listen on the port specified by the PORT environment variable, which defaults to 8080.
EXPOSE 8080

# Command to run your application using Gunicorn with Uvicorn workers
# Adjusted to use 'app.main:app'
# Ensure 'gunicorn' and 'uvicorn' are in your BE/requirements.txt
CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker --threads 8 app.main:app