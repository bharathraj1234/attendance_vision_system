import cv2, os, time
import numpy as np

url = 'http://192.0.0.4:8080/video'  # Correct IP address
size = 2  # Downscaling size for performance
classifier = 'haarcascade_frontalface_default.xml'
image_dir = 'images'

(images, labels, names, id) = ([], [], {}, 0)

# Load training data from images directory
for (subdirs, dirs, files) in os.walk(image_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(image_dir, subdir)
        for filename in os.listdir(subjectpath):
            f_name, f_extension = os.path.splitext(filename)
            if f_extension.lower() not in ['.png', '.jpg', '.jpeg', '.pgm']:
                continue
            path = os.path.join(subjectpath, filename)
            label = id
            images.append(cv2.imread(path, 0))  # Grayscale read
            labels.append(int(label))
        id += 1

(im_width, im_height) = (112, 92)

# Convert images and labels to numpy arrays
(images, labels) = [np.array(lis) for lis in [images, labels]]

# Create LBPH face recognizer and train
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

haar_cascade = cv2.CascadeClassifier(classifier)

# Open video stream
webcam = cv2.VideoCapture(url)  # IP camera or webcam feed
while True:
    rval = False
    while not rval:
        rval, frame = webcam.read()
        if not rval:
            print("Failed to open webcam. Retrying...")

    startTime = time.time()

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, (gray.shape[1] // size, gray.shape[0] // size))

    # Detect faces
    faces = haar_cascade.detectMultiScale(mini)
    for (x, y, w, h) in faces:
        (x, y, w, h) = [v * size for v in (x, y, w, h)]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        prediction = model.predict(face_resize)

        # Draw rectangle and name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.rectangle(frame, (x, y - 20), (x + 120, y), (0, 255, 255), -3)

        if prediction[1] < 90:
            cv2.putText(frame, f'{names[prediction[0]]} - {prediction[1]:.0f}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            cv2.putText(frame, f'Unknown {prediction[1]:.0f}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Calculate FPS
    endTime = time.time()
    fps = 1 / (endTime - startTime)
    cv2.putText(frame, f'Fps: {int(fps)}', (34, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show frame
    cv2.imshow('Recognition System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
