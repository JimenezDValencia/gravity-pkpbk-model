
import os
import runpy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PK_METRICS_SCRIPT = os.path.join(PROJECT_ROOT, "src", "pk_metrics.py")

if __name__ == "__main__":
    print("Running gravity-modulated PK/PBPK simulations...")
    runpy.run_path(PK_METRICS_SCRIPT, run_name="__main__")
    print("Done.")