import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.convnext import preprocess_input

app = Flask(__name__)
socketio = SocketIO(app)

# Load mô hình đã được huấn luyện
model_path = "/home/trung/final/regnet/convnext/best_model.keras"
model = tf.keras.models.load_model(model_path)

# Load file Haar Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Các class tương ứng với cảm xúc
class_names = ["angry", "happy", "neutral", "sad", "surprise"]

# Hàm tiền xử lý ảnh từ frame khuôn mặt
def preprocess_image_from_face(face, target_size=(224, 224)):
    face_resized = cv2.resize(face, target_size)
    img_array = image.img_to_array(face_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Tiền xử lý theo chuẩn ConvNext
    return img_array

# Hàm dự đoán cảm xúc từ khuôn mặt
def predict_emotion_from_face(face):
    img_array = preprocess_image_from_face(face)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    class_label = class_names[predicted_class[0]]
    return class_label

# Xử lý mỗi frame video từ client
@socketio.on('video_frame')
def handle_video_frame(data):
    # Giải mã frame từ base64
    img_data = data['image']
    np_arr = np.frombuffer(np.fromstring(img_data, np.uint8), np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Phát hiện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        class_label = predict_emotion_from_face(face)

        # Vẽ khung và nhãn lên frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, class_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Encode frame và gửi lại client
    _, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = buffer.tobytes()
    emit('emotion_result', {'image': frame_encoded})

# Route trang chủ
@app.route('http://emotion46')
def home():
    return render_template('index.html')  # Trả về tệp index.html

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
