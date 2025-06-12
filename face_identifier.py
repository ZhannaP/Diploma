import numpy as np
from face_database import FaceDatabase

def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class FaceIdentifier:
    def __init__(self, db_path="face_db.sqlite", threshold=0.6):
        self.db = FaceDatabase(db_path)
        self.threshold = threshold  # чим ближче до 1 — тим краще

    def identify(self, embedding):
        known = self.db.get_all_embeddings()
        best_score = -1
        best_name = "Unknown"

        for name, known_embedding in known:
            score = cosine_similarity(embedding, known_embedding)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= self.threshold:
            return best_name, best_score
        else:
            return "Unknown", best_score

    def close(self):
        self.db.close()
