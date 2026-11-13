"""Regression checks for selectively imported ASR postprocessing and VAD."""
import sys,inspect,re
import tempfile,types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'s2s_edge'))
from s2s_edge.textnorm import vi_punct
from s2s_edge.audio import segment,SR
from s2s_edge.conversation import restore_vi
from s2s_edge.backends import OrtGraph


class AsrIntegrationTests(unittest.TestCase):
    def test_upstream_examples(self):
        self.assertEqual(vi_punct('hẹn cuối là thứ sáu đúng không'),'Hẹn cuối là thứ Sáu đúng không?')
        self.assertEqual(vi_punct('tôi rất hài lòng bao nhiêu tiền'),'Tôi rất hài lòng. Bao nhiêu tiền?')
        self.assertEqual(vi_punct('xin chào tôi là nam'),'Xin chào! Tôi là nam.')

    def test_postprocess_capu_model(self):
        with patch('s2s_edge.punct_vi.get',return_value=lambda t:'Tôi rất hài lòng bao nhiêu tiền.'):
            self.assertEqual(restore_vi('tôi rất hài lòng bao nhiêu tiền'),'Tôi rất hài lòng. Bao nhiêu tiền?')

    def test_numeric_content_not_changed(self):
        for text in ['nhiệt độ 37.5 độ c','gọi số 0912345678','mã vn247 lúc 14:45 ngày 21/12/2027']:
            self.assertEqual(re.findall(r'\d+',vi_punct(text)),re.findall(r'\d+',text))

    def test_negative_statement(self):
        self.assertTrue(vi_punct('tôi không biết khi nào').endswith('.'))
        self.assertTrue(vi_punct('không sao').endswith('.'))

    def test_numeric_zero_is_not_question_particle(self):
        self.assertEqual(vi_punct('Gọi số không.'),'Gọi số không.')
        self.assertEqual(vi_punct('Năm, hai nghìn, không.'),'Năm, hai nghìn, không.')
        self.assertEqual(vi_punct('Anh có số không.'),'Anh có số không?')

    def test_vad_and_silence(self):
        self.assertEqual(inspect.signature(segment).parameters['min_sil_ms'].default,320)
        self.assertEqual(segment(np.zeros(SR,dtype=np.float32)),[])
        t=np.arange(SR)/SR
        voice=(0.15*np.sin(2*np.pi*220*t)).astype(np.float32)
        wav=np.concatenate([np.zeros(3200),voice,np.zeros(9600),voice,np.zeros(3200)]).astype(np.float32)
        parts=segment(wav);self.assertEqual(len(parts),2)
        self.assertTrue(all(0<=a<b<=len(wav) for a,b in parts))

    def test_ort_optimizer_failure_retries_without_optimization(self):
        levels=[]
        def session(path, options, providers):
            levels.append(options.graph_optimization_level)
            if len(levels)==1:raise RuntimeError('unsupported optimizer')
            return types.SimpleNamespace(get_inputs=lambda:[])
        fake=types.SimpleNamespace(SessionOptions=types.SimpleNamespace,
            GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_BASIC=1,ORT_DISABLE_ALL=0),InferenceSession=session)
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules,{'onnxruntime':fake}):
            path=Path(folder)/'small.onnx';path.write_bytes(b'fake optimizer test')
            OrtGraph(str(path))
        self.assertEqual(levels,[1,0])


if __name__=='__main__':unittest.main()
