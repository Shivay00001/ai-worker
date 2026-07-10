#!/usr/bin/env python3
import sys
import os

# Add the src directory to PYTHONPATH so imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ai_worker.cli import cli

if __name__ == "__main__":
    cli()
