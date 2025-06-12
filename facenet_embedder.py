from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
import cv2

class FaceEmbedder:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def preprocess_face(self, face_img):
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (160, 160))
        face_tensor = torch.tensor(face_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        face_tensor = (face_tensor - 127.5) / 128.0
        return face_tensor.to(self.device)

    def get_embedding(self, face_img):
        face_tensor = self.preprocess_face(face_img)
        with torch.no_grad():
            embedding = self.model(face_tensor)
        return embedding.squeeze().cpu().numpy() 
