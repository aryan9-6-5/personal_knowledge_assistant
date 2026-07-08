import sqlite3
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they do not exist."""
    logger.info("Initializing database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Conversations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    # Messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        sources TEXT, -- JSON string representing source documents
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    );
    """)
    
    # Documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        chunks INTEGER NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        file_path TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

# User operations
def create_user(username: str, hashed_password: str) -> Optional[int]:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
            (username, hashed_password)
        )
        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# Document operations
def add_document(doc_id: str, user_id: int, name: str, chunks: int, status: str, created_at: str, file_path: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (id, user_id, name, chunks, status, created_at, file_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (doc_id, user_id, name, chunks, status, created_at, file_path)
    )
    conn.commit()
    conn.close()

def update_document_status(doc_id: str, user_id: int, status: str, chunks: int = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if chunks is not None:
        cursor.execute(
            "UPDATE documents SET status = ?, chunks = ? WHERE id = ? AND user_id = ?",
            (status, chunks, doc_id, user_id)
        )
    else:
        cursor.execute(
            "UPDATE documents SET status = ? WHERE id = ? AND user_id = ?",
            (status, doc_id, user_id)
        )
    conn.commit()
    conn.close()

def get_user_documents(user_id: int) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_document(doc_id: str, user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_user_document(doc_id: str, user_id: int) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0

def clear_user_documents(user_id: int) -> List[str]:
    """Delete all documents for a user, return their file paths so they can be unlinked from disk."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM documents WHERE user_id = ? AND file_path IS NOT NULL", (user_id,))
    paths = [r["file_path"] for r in cursor.fetchall()]
    cursor.execute("DELETE FROM documents WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return paths

# Conversation operations
def create_conversation(conv_id: str, user_id: int, title: str) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
        (conv_id, user_id, title)
    )
    conn.commit()
    conn.close()
    return conv_id

def get_user_conversations(user_id: int) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM conversations WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def conversation_exists(conv_id: str, user_id: int) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id))
    row = cursor.fetchone()
    conn.close()
    return row is not None

def add_message(msg_id: str, conv_id: str, role: str, content: str, sources: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (id, conversation_id, role, content, sources) VALUES (?, ?, ?, ?, ?)",
        (msg_id, conv_id, role, content, sources)
    )
    conn.commit()
    conn.close()

def get_conversation_history(conv_id: str) -> List[Dict[str, str]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC", (conv_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]

def clear_user_conversations(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
