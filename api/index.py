"""Vercel serverless entrypoint for FastAPI app."""

import os
import sys

BASE_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.join(BASE_DIR, "..", "backend")

# Ensure backend/app is importable when running on Vercel.
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from app.main import app
