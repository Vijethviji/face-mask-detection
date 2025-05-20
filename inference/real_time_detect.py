import cv2
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
from torch import nn

# === Load the trained model (full object) ===
model_path = r"C:\Users\Vijeth B V\OneDrive\Desktop\abhijna\models\mask_detector_model.pth"
model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
model.eval()

# === Transformation for input frames ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Labels ===
labels = ["Mask", "No Mask"]

# === Load OpenCV face detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Start Webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam could not be opened")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            img = transform(face_img).unsqueeze(0)
            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output, 1)
                label = labels[predicted.item()]

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' key pressed. Exiting...")
            break

except KeyboardInterrupt:
    print("\n[INFO] KeyboardInterrupt received. Exiting gracefully...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam and windows closed.")
