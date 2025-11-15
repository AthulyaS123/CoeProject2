FROM python:3.11

# Install the same TensorFlow + Flask versions as your venv
RUN pip install --no-cache-dir "tensorflow==2.20.0" "Flask==3.1.2"

# Workdir inside the container
WORKDIR /app

# Copy your API code and trained model into the image
COPY api.py /app/api.py
COPY damage.keras /app/damage.keras

# Expose Flask port
EXPOSE 5000

# Run the server
CMD ["python", "api.py"]
