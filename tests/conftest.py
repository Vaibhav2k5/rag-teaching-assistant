import sys
from pathlib import Path

# make "import rag_ta" work when running pytest from the project folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
