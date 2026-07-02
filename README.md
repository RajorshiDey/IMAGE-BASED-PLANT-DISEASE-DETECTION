# 🌿 Plant Disease Classification API

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/MobileNetV2-CNN-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge">
</p>

<p align="center">
<b>An AI-powered plant disease classification API built with TensorFlow and FastAPI.</b>

Upload a plant leaf image and receive an instant disease prediction with confidence score using a pre-trained MobileNetV2 model.
</p>

---

# Plant Disease Classification API

A lightweight REST API for detecting plant diseases from leaf images. The application loads a pre-trained **MobileNetV2** model at startup and predicts one of **38 plant disease classes** through an easy-to-use FastAPI endpoint.

---

## Features

- Image-based plant disease detection
- 38 plant disease categories
- Confidence score prediction
- FastAPI REST API
- TensorFlow MobileNetV2 model
- Automatic model loading at startup
- CORS enabled for frontend integration

---

## Tech Stack

- Python
- TensorFlow
- FastAPI
- Uvicorn
- Pillow
- NumPy

---

## Project Structure

```text
Plant-Disease-Classification/
│
├── main.py                          # FastAPI application
├── plant_disease_model_final.keras  # Trained TensorFlow model
├── requirements.txt                 # Project dependencies
└── README.md
```

---

## Supported Diseases

The model can classify **38 classes** across multiple crops, including:

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

Each prediction returns:

- Disease name
- Confidence score

---

## Installation

Clone the repository:

```bash
git clone https://github.com/RajorshiDey/IMAGE-BASED-PLANT-DISEASE-DETECTION.git

cd IMAGE-BASED-PLANT-DISEASE-DETECTION
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Check API status |
| POST | `/predict` | Predict plant disease from an uploaded image |

---

## Example Request

Using cURL:

```bash
curl -X POST \
  "http://127.0.0.1:8000/predict" \
  -F "file=@leaf.jpg"
```

---

## Example Response

```json
{
  "filename": "leaf.jpg",
  "prediction": "Tomato___Early_blight",
  "confidence": 98.74
}
```

---

## How It Works

1. Upload a plant leaf image.
2. The image is resized to **224 × 224** pixels.
3. MobileNetV2 preprocesses the image.
4. The trained TensorFlow model predicts the disease class.
5. The API returns the predicted disease and confidence score.

---

## Future Enhancements

- Disease treatment recommendations
- Batch image prediction
- Grad-CAM visualization
- Model optimization with TensorFlow Lite
- Web dashboard
- Mobile application

---

## Contributing

Contributions are welcome! Feel free to fork the repository, improve the project, and submit a pull request.

---

## License

This project is intended for educational and research purposes.

---

## Author

**Rajorshi Dey**

If you found this project useful, consider giving it a ⭐
