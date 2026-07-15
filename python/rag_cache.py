# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the GNU GPLv3 license.  
# For a copy, see <https://www.gnu.org/licenses/>.
# 

import sqlite3
import numpy as np

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class RAGCache:
    def __init__(self, db_path="rag_cache.db", d_model=256):
        self.db_path = db_path
        self.d_model = d_model

        # Initialize SQLite DB
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

        # Initialize FAISS index (L2 distance)
        if HAS_FAISS:
            self.index = faiss.IndexFlatL2(self.d_model)
        else:
            print("Warning: FAISS not installed. Semantic caching will be disabled.")
            self.index = None

        self.metadata = {}  # Map FAISS index ID to SQLite metadata ID
        self._current_idx = 0

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_hash TEXT UNIQUE,
                transcript TEXT,
                state_vector BLOB
            )
        """)
        self.conn.commit()

    def store_state(
        self, audio_hash: str, state_vector: np.ndarray, transcript: str = ""
    ):
        if not HAS_FAISS:
            return

        # Store in SQLite
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO cache (audio_hash, transcript, state_vector) VALUES (?, ?, ?)",
            (audio_hash, transcript, state_vector.tobytes()),
        )
        self.conn.commit()
        db_id = cursor.lastrowid

        # Add to FAISS index
        state_vector_2d = state_vector.reshape(1, -1).astype(np.float32)
        self.index.add(state_vector_2d)

        # Map FAISS ID to SQLite ID
        self.metadata[self._current_idx] = db_id
        self._current_idx += 1

    def search_similar(self, query_vector: np.ndarray, top_k: int = 1):
        if not HAS_FAISS or self.index.ntotal == 0:
            return []

        query_vector_2d = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_vector_2d, top_k)

        results = []
        cursor = self.conn.cursor()
        for idx in indices[0]:
            if idx in self.metadata:
                db_id = self.metadata[idx]
                cursor.execute(
                    "SELECT audio_hash, transcript FROM cache WHERE id = ?", (db_id,)
                )
                row = cursor.fetchone()
                if row:
                    results.append({"audio_hash": row[0], "transcript": row[1]})
        return results

    def close(self):
        self.conn.close()
