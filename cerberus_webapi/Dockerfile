FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the API requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY . .

# The project root should be mounted as a volume
# docker run -v /path/to/cerberus-project:/app/cerberus cerberus-api

# Create a symbolic link from /app/cerberus to the expected parent directory
# This is done at runtime to support volume mounting
CMD ln -sf /app/cerberus /app/.. && \
    python app.py

EXPOSE 5000