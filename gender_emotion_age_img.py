import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load face classifier, emotion detection model, gender, and age detection models
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('model.h5')
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

# Load image
image_path = 'Your_image.jpg'
frame = cv2.imread(image_path)

padding = 20

# Convert frame to grayscale for face detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the frame
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    # Extract face region
    face_roi = gray[y:y+h, x:x+w]

    # Resize face for emotion detection
    roi = cv2.resize(face_roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # Predict emotion
    preds = emotion_model.predict(roi)[0]
    emotion_label = emotion_labels[preds.argmax()]

    # Draw label on the frame
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

    # Gender and age estimation
    face = frame[max(0, y - padding):min(y + h + padding, frame.shape[0] - 1),
                 max(0, x - padding):min(x + w + padding, frame.shape[1] - 1)]
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Gender prediction
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    # Age prediction
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    label = "{},{},{}".format(gender, age, emotion_label)
    cv2.putText(frame, label, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

# Display frame
cv2.imshow('Emotion, Age, and Gender Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
