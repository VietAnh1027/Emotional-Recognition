import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms, models

cap = cv2.VideoCapture(0)

# Khai báo model nhận diện khuôn mặt
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face.FaceDetection(min_detection_confidence=0.65)

# Khai báo model phân biệt cảm xúc
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
model.load_state_dict(torch.load("models/efficientnet_b0_best_latest.pth", weights_only=False))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']
# labels_temp = ['_','_','_','Happy','Neutral','_','_']

# Khai báo tiền xử lý ảnh cho model cảm xúc
my_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(rgb_frame)

    if result.detections:
        for detection in result.detections:
            # Vẽ bounding box lên ảnh
            mp_draw.draw_detection(frame, detection)
            # Vẽ độ tin cậy
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x1, y1 = int(bbox.xmin * iw), int(bbox.ymin * ih)
            w, h = int(bbox.width * iw), int(bbox.height * ih)
            face_img = frame[y1:y1+h,x1:x1+w]
            face_img = my_transform(face_img).unsqueeze(0)
            with torch.no_grad():
                face_img = face_img.to(device)
                output = model(face_img)
                _, pred = torch.max(output, 1)
            score = detection.score[0]
            text = f"Confident: {score*100:.1f}%Emotion: {labels[pred.item()]}"
            text2 = f"Emotion: {labels[pred.item()]}"
            cv2.putText(frame, text2, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX,  0.6, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()