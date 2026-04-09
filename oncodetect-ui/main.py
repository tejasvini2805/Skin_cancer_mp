from pathlib import Path
import sys

# Ensure the parent directory of the workspace is on sys.path so `sc_app` can be imported
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

try:
    from sc_app.main import app  # type: ignore
except Exception as e:
    raise ImportError(f"Failed to import `app` from sc_app.main: {e}")
