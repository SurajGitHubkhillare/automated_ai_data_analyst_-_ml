import sqlite3
import bcrypt
from datetime import datetime
import pandas as pd
import os

DB_FILE = 'app_data.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    # Create login_logs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            login_time TEXT NOT NULL
        )
    ''')
    conn.commit()

    # Create default admin if not exists
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    admin = c.fetchone()
    if not admin:
        pw_hash = hash_password("admin")
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", ("admin", pw_hash, "Admin"))
        conn.commit()
        
    conn.close()

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def add_user(username, password, role="User"):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        pw_hash = hash_password(password)
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", (username, pw_hash, role))
        conn.commit()
        return True, "User created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def delete_user(username):
    if username == 'admin':
        return False, "Cannot delete default admin."
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        return True, "User deleted successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    record = c.fetchone()
    
    if record:
        stored_hash = record[0]
        role = record[1]
        if verify_password(password, stored_hash):
            # Log the login
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO login_logs (username, login_time) VALUES (?, ?)", (username, current_time))
            conn.commit()
            conn.close()
            return True, role
    
    conn.close()
    return False, None

def get_all_users():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT id, username, role FROM users", conn)
    conn.close()
    return df

def get_login_logs():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM login_logs ORDER BY login_time DESC", conn)
    conn.close()
    return df
