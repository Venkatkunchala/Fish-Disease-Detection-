import os
import sys  # Import sys to configure encoding
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image
 
# Configure UTF-8 encoding to avoid Unicode errors
sys.stdout.reconfigure(encoding='utf-8')

def train_model():
    base_dir = 'C:\\Users\\venka\\OneDrive\\Desktop\\git'
    train_dir = os.path.join(base_dir, 'FishML\\Dataset.csv\\training_set')
    validation_dir = os.path.join(base_dir, 'FishML\\Dataset.csv\\validation_set')

    img_width, img_height = 150, 150
    batch_size = 20

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory does not exist: {train_dir}")
    if not os.path.exists(validation_dir):
        raise FileNotFoundError(f"Validation directory does not exist: {validation_dir}")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=RMSprop(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=50,
        verbose=2
    )

    model.save(os.path.join(base_dir, 'FishML\\model_trained.h5'))
    return model

def load_trained_model(model_path='C:\\Users\\venka\\OneDrive\\Desktop\\git\\FishML\\model_trained.h5'):
    if not os.path.exists(model_path):
        return train_model()
    else:
        return tf.keras.models.load_model(model_path)

def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.convert('RGB')
    img = img.resize((150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def calculate_optimal_threshold(model, healthy_img_path, infected_img_path):
    healthy_img = preprocess_image(healthy_img_path)
    infected_img = preprocess_image(infected_img_path)

    healthy_pred = model.predict(healthy_img)
    infected_pred = model.predict(infected_img)

    threshold = (healthy_pred[0][0] + infected_pred[0][0]) / 2
    return threshold

def main():
    st.title("Fish Disease Detection App")
    model = load_trained_model()

    # Use a fixed threshold of 0.6
    fixed_threshold = 0.6
    st.write(f"Using Fixed Threshold: {fixed_threshold:.2f}")

    image_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if image_file is not None:
        st.image(image_file, use_column_width=True)
        processed_image = preprocess_image(image_file)
        predictions = model.predict(processed_image)

        if predictions[0][0] <= fixed_threshold:
            st.write(f"Prediction: This fish is healthy (Prediction Score: {predictions[0][0]:.4f})")
        else:
            st.write(f"Prediction: This fish is diseased (Prediction Score: {predictions[0][0]:.4f})")

if __name__ == "__main__":
    main()