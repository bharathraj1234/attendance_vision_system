# Import necessary modules
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
from aiortc import RTCPeerConnection, RTCSessionDescription
import cv2
import numpy as np
import os
import time
import logging
import uuid
import asyncio

# Flask app
app = Flask(__name__, static_url_path='/static')

url = 'http://192.0.0.4:8080/video'
size = 2  # Change to 4 to speed up processing, but at the cost of accuracy
classifier = 'haarcascade_frontalface_default.xml'
image_dir = 'images'

# Face recognition setup
print("Face Recognition Starting ...")
(images, labels, names, id) = ([], [], {}, 0)

# Load training images
for (subdirs, dirs, files) in os.walk(image_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(image_dir, subdir)
        for filename in os.listdir(subjectpath):
            f_name, f_extension = os.path.splitext(filename)
            if f_extension.lower() not in ['.png', '.jpg', '.jpeg', '.gif', '.pgm']:
                print("Skipping " + filename + ", wrong file type")
                continue
            path = os.path.join(subjectpath, filename)
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(im_width, im_height) = (120, 102)

# Convert images and labels to NumPy arrays
(images, labels) = [np.array(lis) for lis in [images, labels]]

# Train the recognizer
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Load the Haar cascade
haar_cascade = cv2.CascadeClassifier(classifier)

# Function to generate video frames and perform face recognition
def generate_frames():
    camera = cv2.VideoCapture(url)
    while True:
        rval, frame = camera.read()
        if not rval:
            print("Failed to open webcam. Trying again...")
            break

        # Flip the frame
        frame = cv2.flip(frame, 1)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the frame to speed up detection
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        # Detect faces
        faces = haar_cascade.detectMultiScale(mini)
        for i in range(len(faces)):
            face_i = faces[i]

            # Get coordinates and scale them back
            (x, y, w, h) = [v * size for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))

            # Predict the face
            prediction = model.predict(face_resize)
            start = (x, y)
            end = (x + w, y + h)

            # Draw a bounding box around the face
            cv2.rectangle(frame, start, end, (0, 255, 0), 3)
            cv2.rectangle(frame, (start[0], start[1] - 20), (start[0] + 120, start[1]), (0, 255, 255), -3)

            # Add the name or 'Unknown' label
            if prediction[1] < 90:
                cv2.putText(frame, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                print('%s - %.0f' % (names[prediction[0]], prediction[1]))
            else:
                cv2.putText(frame, "Unknown %.0f" % prediction[1], (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                print("Unknown -", prediction[1])

        # Encode the frame for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')

