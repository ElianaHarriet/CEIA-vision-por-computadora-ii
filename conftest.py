import sys
from pathlib import Path

DAGS_DIR = Path(__file__).parent / "airflow" / "dags"
if str(DAGS_DIR) not in sys.path:
    sys.path.insert(0, str(DAGS_DIR))