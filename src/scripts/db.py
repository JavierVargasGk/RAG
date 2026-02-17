from dotenv import load_dotenv
import os
import sys

load_dotenv()

def get_connection_string():
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname = os.getenv("DB_NAME")
    
    
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


