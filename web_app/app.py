from flask import Flask, render_template, Response
import cv2
import torch
from torchvision import transforms

app = Flask(__name__)

# Load your model
model_path = r"C:\Users\Vijeth B V\OneDrive\Desktop\abhijna\models\mask_detector_model.pth"
model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

labels = ["Mask", "No Mask"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

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

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame as multipart JPEG to stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
