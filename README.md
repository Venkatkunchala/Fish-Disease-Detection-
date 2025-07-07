**🐟 Fish Disease Detection Web Application**
A deep learning-powered web app that detects whether a fish is healthy or diseased using Convolutional Neural Networks (CNN) and an interactive Streamlit interface.

**🧾 Project Overview**
This project provides an intuitive solution for identifying fish diseases through image classification. It is designed to assist fish farmers, researchers, and marine biologists by using deep learning to automate diagnosis from fish images.

**🧠 Model Highlights**
Image Input Size: 150 × 150 pixels (RGB)

**Model Architecture:**

3 Convolutional Layers with ReLU activation

MaxPooling Layers for dimensionality reduction

Dense Layers for classification

Output Layer with Sigmoid Activation (binary classification)

Loss Function: Binary Crossentropy

**Optimizer:** RMSprop

**Prediction Output: **Score between 0 and 1

≤ 0.6 → Healthy

> 0.6 → Diseased

**⚙️ Key Features**
📤 Upload your own fish image via the web interface

📸 Real-time prediction using a trained CNN model

🔁 Auto-retraining if the model file is missing

🔧 Advanced image preprocessing and augmentation

✅ Clean and user-friendly UI with Streamlit

**🗂️ Project Structure Overview**
Dataset.csv/
├─ training_set/ → Healthy/, Infected/
└─ validation_set/ → Healthy/, Infected/

model_trained.h5 – Trained CNN model

app.py – Streamlit application script

README.md – Project documentation

**🚀 How It Works**
Upload a .jpg, .png, or .jpeg fish image through the app.

The image is resized and preprocessed to match model input dimensions.

The image is passed into the trained CNN model for prediction.

A prediction score is returned:

Healthy if the score is ≤ 0.6

Diseased if the score is > 0.6

The result is displayed with the prediction confidence.

📊 Sample Prediction Output
✅ Healthy – Prediction Score: 0.4231

⚠️ Diseased – Prediction Score: 0.8114

**🔁 Model Training & Retraining**
The model is trained using augmented image data to improve accuracy and generalization.

If model_trained.h5 is not found, the application automatically retrains using the dataset in training_set/ and validation_set/.

**👨‍💻 Author**
Venkat Kunchala
📧 Email: venkatkunchala095@gmail.com
