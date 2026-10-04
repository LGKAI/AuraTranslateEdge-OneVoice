
import warnings
from contextlib import contextmanager
from packaging import version
import torch

TORCH_VERSION = version.parse(torch.__version__.split("+")[0])

@contextmanager
def torch_autocast(device_type="cuda", **kwargs):
    if TORCH_VERSION >= version.parse("2.3.0"):
        with torch.amp.autocast(device_type=device_type, **kwargs):
            yield
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(**kwargs):
                yield
