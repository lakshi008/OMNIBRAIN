"""
OmniBrain Member 4 Integration & Quality Assurance Test Suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path for test discovery and direct execution
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
