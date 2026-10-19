
import os
import sqlite3
from datetime import datetime
import numpy as np


class Database:
    """Stores registered faces and entry/exit events in SQLite."""

    def __init__(self, path):
        folder = os.path.dirname(path)

        if folder:
            os.makedirs(folder, exist_ok=True)

        self.conn = sqlite3.connect(path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                face_id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_seen TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                FOREIGN KEY (face_id) REFERENCES faces(face_id)
            )
        """)

        self.conn.commit()

    def add_face(self, embedding):
        """Register a new face. Returns the new face_id."""
        ts = datetime.now().isoformat(timespec="seconds")

        embedding = np.asarray(embedding, dtype=np.float32)

        cur = self.conn.execute(
            "INSERT INTO faces (first_seen, embedding) VALUES (?, ?)",
            (ts, embedding.tobytes())
        )

        self.conn.commit()
        return cur.lastrowid

    def get_all_faces(self):
        """Return {face_id: embedding} for every registered face."""
        rows = self.conn.execute(
            "SELECT face_id, embedding FROM faces"
        ).fetchall()

        return {
            fid: np.frombuffer(blob, dtype=np.float32).copy()
            for fid, blob in rows
        }

    def add_event(self, face_id, event_type, image_path=None):
        """Store one entry or exit event."""
        if event_type not in ("entry", "exit"):
            raise ValueError("event_type must be 'entry' or 'exit'")

        ts = datetime.now().isoformat(timespec="seconds")

        self.conn.execute(
            """
            INSERT INTO events (face_id, event_type, timestamp, image_path)
            VALUES (?, ?, ?, ?)
            """,
            (face_id, event_type, ts, image_path)
        )

        self.conn.commit()

    def get_unique_count(self):
        """Return the number of registered face records."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM faces"
        ).fetchone()[0]

    def close(self):
        """Close the database connection."""
        self.conn.close()
