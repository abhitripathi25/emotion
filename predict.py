import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("mobile_net.h5")

# Define class names
class_names = ["control", "gore", "pornpics"]  # Update with your class names

# Function to predict and annotate images or video frames
def predict_and_annotate(file_path, result_folder):
    # Load the image or video frame
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(file_path)
        if img is None:
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0 
        img_array = np.expand_dims(img_resized, axis=0)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]

        # Save annotated image if the predicted class is "pornpics" or "gore"
        if predicted_class_name in ["pornpics", "gore"]:
            # Annotate image
            cv2.putText(img, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Save annotated image to result folder
            result_path = os.path.join(result_folder, os.path.basename(file_path))
            cv2.imwrite(result_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Annotated image saved to {result_path}")

    elif file_path.lower().endswith(('.mp4', '.avi')):
        cap = cv2.VideoCapture(file_path)
        frame_count = 0
        prev_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Check time for every 10 seconds
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if current_time - prev_time >= 10:
                prev_time = current_time

                frame_resized = cv2.resize(frame, (224, 224)) / 255.0
                frame_array = np.expand_dims(frame_resized, axis=0)
                predictions = model.predict(frame_array)
                predicted_class_index = np.argmax(predictions)
                predicted_class_name = class_names[predicted_class_index]

                # Save annotated frame if the predicted class is "pornpics" or "gore"
                if predicted_class_name in ["pornpics", "gore"]:
                    # Annotate frame
                    cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Save annotated frame to result folder
                    result_path = os.path.join(result_folder, f"{os.path.basename(file_path)}_frame{frame_count}.jpg")
                    cv2.imwrite(result_path, frame)
                    print(f"Annotated frame {frame_count} saved to {result_path}")

        cap.release()

# Folder containing files
input_folder = r'C:/Users/DELL/OneDrive/Desktop/dataset/input'

# Folder to save annotated images and frames
result_folder = r'C:/Users/DELL/OneDrive/Desktop/dataset/result'

# Iterate through files in the input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    if os.path.isfile(file_path):
        # Check if the file is an image or a video
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi')):
            predict_and_annotate(file_path, result_folder)
