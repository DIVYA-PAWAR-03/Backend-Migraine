"""Vercel serverless entrypoint for FastAPI app."""

import os
import sys

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(BASE_DIR, "..")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app.main import app
