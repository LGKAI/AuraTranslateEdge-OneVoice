# -*- coding: utf-8 -*-
"""Probe: Melo TTS as ONE graph = graph A (static front + BERT + expansion) -> VITS g5. Input: UTF-8 bytes [1,112]; outputs audio, y_len."""
import onnx, numpy as np, onnxruntime as ort, os
from onnx import compose
a = onnx.load("work/graphA.onnx"); v = onnx.load("work/melo_N96_Y384_g5.onnx")
a = compose.add_prefix(a, "A_"); v = compose.add_prefix(v, "V_")
io = [("A_output_0", "V_x"), ("A_output_1", "V_tone"), ("A_output_2", "V_lang"), ("A_output_4", "V_bert"), ("A_output_3", "V_x_len")]
m = compose.merge_models(a, v, io_map=io, outputs=["V_audio", "V_y_len"])
m.graph.input[0].name  # A_text_bytes
for n in m.graph.node:
    n.input[:] = ["text_bytes" if x == "A_text_bytes" else x for x in n.input]
    n.output[:] = ["output_0" if x == "V_audio" else "output_1" if x == "V_y_len" else x for x in n.output]
m.graph.input[0].name = "text_bytes"; m.graph.output[0].name = "output_0"; m.graph.output[1].name = "output_1"
del m.graph.value_info[:]
onnx.save(m, "work/melo_one.onnx"); print("MB", os.path.getsize("work/melo_one.onnx") / 1e6, "nodes", len(m.graph.node))
d = np.load("work/graphA_inputs.npz")["text_bytes"]; np.savez("work/melo_one_inputs.npz", text_bytes=d)
s = ort.InferenceSession("work/melo_one.onnx", providers=["CPUExecutionProvider"]); r = s.run(None, {"text_bytes": d[0]})
print("ORT ok", r[0].shape, r[1])
