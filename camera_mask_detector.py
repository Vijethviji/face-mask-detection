import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/mask_detector_model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224)) / 255.0
        face = np.expand_dims(face, axis=0)
        pred = model.predict(face)
        label = "Mask" if np.argmax(pred) == 0 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()