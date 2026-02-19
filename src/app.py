import os

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    target_path = os.path.join("src", "scripts", "search.py")
    run_chainlit(target_path)
