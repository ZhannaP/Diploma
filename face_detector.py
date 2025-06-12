import cv2
import os

class FaceDetector:
    def __init__(self, scaleFactor=1.1, minNeighbors=5):
        # Завантаження каскаду для детекції
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors
        )
        return faces 

    def draw_faces(self, frame, detections):
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det[:4])
            confidence = float(det[4]) if len(det) >= 5 else 0.0

            label = f"Face #{i+1} ({confidence:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame


    def draw_faces(self, frame, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame

    def extract_faces(self, frame, faces, target_size=(160, 160)):
        face_images = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, target_size)
            face_images.append(face_resized)
        return face_images
