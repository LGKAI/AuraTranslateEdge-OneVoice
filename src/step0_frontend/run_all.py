"""Run the full Step-0 front-end test suite end-to-end:
mix_noise -> test_vad -> test_denoise -> test_beamform.
Results land in outputs/*.csv (numbers for Technical Proposal SS4.2/SS4.4)
and outputs/{vad,denoise,beamform}/ (wav files to listen to).
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["mix_noise.py", "test_vad.py", "test_denoise.py", "test_beamform.py"]


def main():
    for step in STEPS:
        print(f"\n{'=' * 70}\n>>> RUNNING {step}\n{'=' * 70}")
        ret = subprocess.call([sys.executable, os.path.join(HERE, step)])
        if ret != 0:
            print(f"[run_all] {step} exited with code {ret} -- continuing anyway")
    print("\n[run_all] Done. See outputs/*.csv for RTF/PESQ/STOI numbers, "
          "and outputs/{vad,denoise,beamform}/ to listen to results.")


if __name__ == "__main__":
    main()
