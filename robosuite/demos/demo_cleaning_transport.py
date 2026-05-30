"""
PlaneFollow cleaning / policy-transportation smoke test.

From the robosuite repo root::

    python robosuite/demos/demo_cleaning_transport.py --surface-type flat --execute
"""

import runpy
import sys
from pathlib import Path

# policy_transportation is a sibling project in the user's workspace
_PT_ROOT = Path(__file__).resolve().parents[3] / "policy_transportation"
if _PT_ROOT.is_dir() and str(_PT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PT_ROOT))

_SCRIPT = _PT_ROOT / "sim" / "cleaning_experiment_plane_follow.py"
if not _SCRIPT.is_file():
    raise FileNotFoundError(
        f"Expected policy_transportation script at {_SCRIPT}. "
        "Clone policy_transportation next to robosuite or set PYTHONPATH."
    )

runpy.run_path(str(_SCRIPT), run_name="__main__")
