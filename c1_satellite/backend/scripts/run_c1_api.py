#!/usr/bin/env python3
"""Run the FloodRisk C1 Satellite API server using Uvicorn.

Usage from repository root:
  python c1_satellite/scripts/run_c1_api.py
  python c1_satellite/scripts/run_c1_api.py --port 8000 --reload
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import uvicorn

# run_c1_api.py is at c1_satellite/backend/scripts/
# parents[1] = c1_satellite/backend/  (contains main.py, improvements/, local_data/, outputs/…)
# parents[2] = c1_satellite/
C1_BACKEND = Path(__file__).resolve().parents[1]   # c1_satellite/backend/
C1_ROOT    = Path(__file__).resolve().parents[2]   # c1_satellite/
# Add backend/ to sys.path so uvicorn finds main.py and improvements package
sys.path.insert(0, str(C1_BACKEND))


def main():
    parser = argparse.ArgumentParser(description="FloodRisk C1 API Server Launcher")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"Starting FloodRisk Sri Lanka C1 API on http://{args.host}:{args.port}")
    print(f"Swagger interactive documentation: http://{args.host}:{args.port}/docs")
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
