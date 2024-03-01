import os
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import zipfile
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("mobile_net.h5")

# Define class names
class_names = ["control", "gore", "pornpics"]  # Update with your class names

# Function to check if the filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'zip'}

# Function to predict and annotate images or video frames
def predict_and_annotate(file_path, result_folder, count_dict):
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
            count_dict[predicted_class_name] += 1

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
                    count_dict[predicted_class_name] += 1

        cap.release()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(zip_path)
        
        # Create a temporary directory for unzipping
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Unzip the uploaded file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Folder to save annotated images and frames
        result_folder = os.path.join(temp_dir, 'result')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        
        # Count dictionary to store the counts
        count_dict = {"pornpics": 0, "gore": 0}
        
        # Iterate through files in the temp folder
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                if os.path.isfile(file_path):
                    # Check if the file is an image or a video
                    if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi')):
                        predict_and_annotate(file_path, result_folder, count_dict)
        
        # Create a zip file containing the annotated images
        annotated_zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated.zip')
        with zipfile.ZipFile(annotated_zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(result_folder):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))
        
        # Remove the temporary directory
        shutil.rmtree(temp_dir)
        
        # Return the annotated zip file and counts
        return jsonify({'message': 'Files uploaded and processed successfully', 'counts': count_dict, 'annotated_zip': annotated_zip_path})
    
    return jsonify({'error': 'Invalid file format'})

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
