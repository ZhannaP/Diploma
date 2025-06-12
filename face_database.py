import sqlite3
import numpy as np

class FaceDatabase:
    def __init__(self, db_path="face_db.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL
            );
        """)
        self.conn.commit()

    def add_embedding(self, name, embedding):
        embedding_bytes = embedding.astype(np.float32).tobytes()
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO embeddings (name, embedding) VALUES (?, ?);", (name, embedding_bytes))
        self.conn.commit()

    def get_all_embeddings(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, embedding FROM embeddings;")
        records = cursor.fetchall()
        results = []
        for name, emb_blob in records:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            results.append((name, emb))
        return results

    def close(self):
        self.conn.close()
