from ultralytics import YOLO
import cv2

class YOLOFaceDetector:
    def __init__(self, model_path="best.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_faces(self, frame):
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy() 
        return detections

    def draw_faces(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def extract_faces(self, frame, detections, target_size=(160, 160)):
        face_images = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            face = frame[y1:y2, x1:x2]
            face_resized = cv2.resize(face, target_size)
            face_images.append(face_resized)
        return face_images
