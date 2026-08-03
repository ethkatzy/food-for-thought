import json
import subprocess
import sys

data = json.load(sys.stdin)
file_path = data.get("tool_input", {}).get("file_path", "")
if file_path.endswith(".py"):
    subprocess.run(["python", "-m", "ruff", "check", "--fix", file_path])
