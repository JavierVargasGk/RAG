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

def delete_file_from_db(filename: str):
    conn_str = get_connection_string()
    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM doc_chunks WHERE filename = %s", (filename,))
                count = cur.fetchone()[0]
                
                if count == 0:
                    print(f"No se encontraron registros para el archivo: {filename}")
                    return

                cur.execute("DELETE FROM doc_chunks WHERE filename = %s", (filename,))
                conn.commit()
                print(f"Éxito: Se eliminaron {count} chunks del archivo '{filename}'.")
                
    except Exception as e:
        print(f"Error al eliminar de la DB: {e}")


