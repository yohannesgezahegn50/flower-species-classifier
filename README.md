# 🌸 Flower Species Classifier

A machine learning web application that identifies flower species from uploaded images using Deep Learning and Transfer Learning.

## 📌 Project Overview

The Flower Species Classifier is an image classification system that predicts the species of a flower from a user-uploaded image.

The model was trained using TensorFlow and EfficientNetB0 through transfer learning and deployed with Streamlit to provide an interactive web interface.

Supported flower classes:

* 🌼 Daisy
* 🌻 Sunflower
* 🌷 Tulip
* 🌹 Rose
* 🌱 Dandelion

---

## 🚀 Features

* Upload flower images directly from your device
* AI-powered flower species prediction
* Confidence score display
* Class probability visualization
* Modern Streamlit user interface
* Fast and lightweight prediction system

---

## 🧠 Machine Learning Model

### Model Architecture

* Base Model: EfficientNetB0
* Framework: TensorFlow / Keras
* Learning Method: Transfer Learning
* Input Size: 224 × 224 × 3

### Training Details

* Dataset: TensorFlow Flowers Dataset
* Number of Classes: 5
* Image Size: 224 × 224
* Epochs: 10
* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy

### Performance

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 99.28% |
| Validation Accuracy | 94.28% |

---

## 🛠 Technologies Used

* Python
* TensorFlow
* Keras
* EfficientNetB0
* NumPy
* Pillow
* Streamlit

---

## 📂 Project Structure

```text
Flower_Classifier/
│
├── app.py
|__ flower_species_classifer.ipynb
├── flower_classifier.keras
├── requirements.txt
└── README.md
```

---

## ⚙ Installation


Navigate into the project folder:

```bash
cd flower-species-classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

---

## 📸 How to Use

1. Open the web application.
2. Upload a flower image.
3. Click the **Predict** button.
4. View the predicted flower species and confidence score.

---

## 🎯 Future Improvements

* Support for additional flower species
* Mobile-optimized interface
* Confidence charts and analytics
* Multi-language support
* Cloud deployment and API integration

---

## 👨‍💻 Author

Developed as a Machine Learning project using TensorFlow, EfficientNetB0, and Streamlit.
