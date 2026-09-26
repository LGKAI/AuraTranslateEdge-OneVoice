"""Run the full Step-1 ASR test suite end-to-end:
fetch_asr_data -> mix_asr_noise -> test_asr_vi -> test_asr_multi.
Results land in outputs/asr_vi_results.csv and outputs/asr_multi_results.csv
(numbers for Technical Proposal SS4.2/SS4.3), plus outputs/asr_vi/*.txt for
per-file reference/hypothesis inspection.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    "fetch_asr_data.py", "mix_asr_noise.py",
    "test_asr_vi.py", "test_asr_multi.py",          # current pick: PhoWhisper + SenseVoice
    "test_asr_zipformer.py", "test_asr_moonshine.py", "test_asr_qwen.py",  # 2026 alt candidates, see step1.md SS12
]


def main():
    for step in STEPS:
        print(f"\n{'=' * 70}\n>>> RUNNING {step}\n{'=' * 70}")
        ret = subprocess.call([sys.executable, os.path.join(HERE, step)])
        if ret != 0:
            print(f"[run_all_asr] {step} exited with code {ret} -- continuing anyway")
    print("\n[run_all_asr] Done. See outputs/asr_*_results.csv for WER/CER/RTF "
          "numbers -- compare asr_vi/asr_multi (current pick) against "
          "asr_zipformer/asr_moonshine/asr_qwen (2026 alt candidates).")


if __name__ == "__main__":
    main()
