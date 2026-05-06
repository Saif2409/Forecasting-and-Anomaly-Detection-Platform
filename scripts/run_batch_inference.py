from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.inference.batch_inference import run_batch_inference


if __name__ == "__main__":
    run_batch_inference()
