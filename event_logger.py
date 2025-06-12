import sqlite3
from datetime import datetime

class EventLogger:
    def __init__(self, db_path="face_db.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                name TEXT NOT NULL,
                score REAL NOT NULL,
                status TEXT NOT NULL,
                source TEXT,
                image_path TEXT
            );
        """)
        self.conn.commit()

    def log(self, name, score, status, source="cam_1", image_path=None):
        timestamp = datetime.now().isoformat(sep=' ', timespec='seconds')
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO access_log (timestamp, name, score, status, source, image_path)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (timestamp, name, score, status, source, image_path))
        self.conn.commit()

    def close(self):
        self.conn.close()
