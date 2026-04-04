"""Vercel serverless entrypoint for FastAPI app."""

import os
import sys

BASE_DIR = os.path.dirname(__file__)
# Support both layouts:
# 1) monorepo: api/ + backend/app
# 2) flattened deploy repo: api/ + app
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "backend"))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

if os.path.isdir(BACKEND_DIR) and BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from app.main import app
