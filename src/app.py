import os
import subprocess
import requests
def get_ollama_endpoint():
    # 1. Check if we are in WSL
    try:
        # This command gets the Gateway IP (the Windows Host)
        window_ip = subprocess.check_output("ip route | grep default | awk '{print $3}'", shell=True).decode().strip()
        print(f"Detected Windows Host IP: {window_ip}")
        # Test if Ollama is reachable on that IP
        test_url = f"http://{window_ip}:11434/api/tags"
        requests.get(test_url, timeout=1)
        return f"http://{window_ip}:11434"
    except:
        # 2. Fallback to localhost (for production or native Linux)
        return "http://localhost:11434"

print(get_ollama_endpoint())