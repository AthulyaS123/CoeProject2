# CoeProject2

# Damage Classification Inference Server

This project provides a Dockerized inference server for a CNN model (Alternate LeNet-5) that classifies post–Hurricane Harvey satellite images as **damage** or **no_damage**! :)

---

API Endpoints
GET /summary: Returns JSON metadata about the model (name, description, version, number of parameters).
POST /inference: Accepts an image file (key: image, sent as multipart/form-data) and returns:
{ "prediction": "damage" }
or
{ "prediction": "no_damage" }

```bash
From the directory containing `docker-compose.yml`, run:
Starting the Server: docker compose up -d
Stopping the Server: docker compose down

Example Requests
Using curl
curl http://localhost:5000/summary

curl -X POST http://localhost:5000/inference \
  -F "image=@/full/path/to/image.jpeg"

Using Python
import requests

summary = requests.get("http://localhost:5000/summary")
print("Summary:", summary.json())

url = "http://localhost:5000/inference"
with open("/full/path/to/image.jpeg", "rb") as f:
    resp = requests.post(url, files={"image": f})
print("Prediction:", resp.json())
