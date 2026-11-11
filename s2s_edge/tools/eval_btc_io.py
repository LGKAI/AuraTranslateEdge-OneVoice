# -*- coding: utf-8 -*-
"""Audio loading shared by the test tools: the organisers' Korean clips are MP3 data in .wav files; libsndfile >= 1.1 reads MP3, else ffmpeg."""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.audio import load_audio


def load_audio_any(path):
    try:
        return load_audio(path)
    except Exception:
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
