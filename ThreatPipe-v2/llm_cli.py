"""Shim so `python llm_cli.py status` works from ThreatPipe-v2/ without -m."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm.cli import main

if __name__ == "__main__":
    main()
