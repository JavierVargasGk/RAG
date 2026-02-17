from dotenv import load_dotenv
import psycopg
import os
import sys

load_dotenv()
def file_exists(filename: str) -> bool:
    conn_str = get_connection_string()
    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM doc_chunks WHERE filename = %s)", 
                    (filename,)
                )
                return cur.fetchone()[0] 
    except Exception as e:
        print(f"Database error while checking file: {e}")
        return False
    
def get_connection_string():
    creds = {
        "user": os.getenv("DB_USER"),
        "pass": os.getenv("DB_PASS"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "name": os.getenv("DB_NAME")
    }
    if not all(creds.values()):
        raise ValueError("Missing database environment variables!")
    
    return f"postgresql://{creds['user']}:{creds['pass']}@{creds['host']}:{creds['port']}/{creds['name']}"


