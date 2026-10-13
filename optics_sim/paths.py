"""Shared project locations; generated artifacts always default to outputs/."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WORKFLOWS_DIR = OUTPUTS_DIR / "workflows"
VALIDATION_DIR = OUTPUTS_DIR / "validation"
