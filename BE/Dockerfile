# BE/Dockerfile

# --- Stage 1: Builder ---
# Use a fuller Python image with build tools for installing dependencies
# 'buster' is the Debian version used by your original slim image
FROM python:3.10.11-buster as builder

# Set working directory for the builder stage
WORKDIR /app

# Install build dependencies required by some Python packages (like gcc, g++)
# Clean up apt cache immediately to keep the image size down even in this stage
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# If you have dependencies like psycopg2-binary, you might need specific dev libraries
# Add necessary libraries here if 'build-essential' isn't enough later.
# Example for psycopg2: RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev gcc python3-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies, which will now include compiling native extensions
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final ---
# Use a slim image for the final runtime environment
FROM python:3.10.11-slim-buster

# Set working directory for the final stage
WORKDIR /app

# Copy the installed packages from the builder stage
# This copies the site-packages directory where pip installed dependencies
# Adjust the path if your python version or environment is different
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# Also copy any executables installed by pip (like 'uvicorn', 'alembic')
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the rest of your backend code
COPY . .

# If your deployed application needs any native *runtime* libraries that are not in 'slim',
# you might need to install them here. This is less common but possible.
# Example if your app directly needs postgres client libs at runtime (most don't):
# RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*


# Command to run your FastAPI application using Uvicorn
# Cloud Run provides the PORT environment variable, your app MUST listen on it.
# Replace 'main:app' with the actual module:variable for your FastAPI app.
# If your app instance is 'my_app' in 'src/api.py', use 'src.api:my_app'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]

# EXPOSE is optional but good practice to indicate which port the container listens on
EXPOSE 8000