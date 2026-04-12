import subprocess
import sys

def run():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True
        )
        with open("install_log.txt", "w", encoding="utf-8") as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
            f.write(f"\n\nRETURN CODE: {result.returncode}")
        print("Log written to install_log.txt")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    run()
