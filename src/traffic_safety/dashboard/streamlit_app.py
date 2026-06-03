"""
Render / production entry point for the Urban Traffic Safety Engine dashboard.

Usage (local):
    streamlit run src/traffic_safety/dashboard/streamlit_app.py

Usage (Render):
    streamlit run src/traffic_safety/dashboard/streamlit_app.py \\
        --server.address 0.0.0.0 --server.port $PORT
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Project layout: <root>/src/traffic_safety/dashboard/streamlit_app.py
_SRC_DIR = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# CSV-only public demo: no PostgreSQL, no on-the-fly dataset rebuilds.
os.environ.setdefault("DASHBOARD_CSV_ONLY", "1")

from traffic_safety.dashboard.app import main  # noqa: E402

if __name__ == "__main__":
    main()
