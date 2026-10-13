import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
