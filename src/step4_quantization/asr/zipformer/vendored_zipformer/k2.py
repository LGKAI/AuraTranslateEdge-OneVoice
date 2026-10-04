
"""Stub thay thế k2 -- chỉ cung cấp đúng 4 hàm scaling.py thực sự dùng (swoosh
activations). Công thức lấy nguyên văn từ SwooshLForward/SwooshRForward trong
chính icefall/egs/librispeech/ASR/zipformer/scaling.py (dùng bởi đường
export ONNX/JIT-trace của chính file đó) -- không phải suy đoán/xấp xỉ.
Không dùng custom CUDA kernel tối ưu bộ nhớ của k2 thật -- backward qua
autograd chuẩn, chậm hơn 1 chút lúc train nhưng numerically identical.
"""
import torch


def swoosh_l_forward(x):
    x_offset = x - 4.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.035


def swoosh_r_forward(x):
    x_offset = x - 1.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.313261687


def swoosh_l(x):
    return swoosh_l_forward(x)


def swoosh_r(x):
    return swoosh_r_forward(x)


def swoosh_l_forward_and_deriv(x):
    with torch.enable_grad():
        x_ = x.detach().requires_grad_(True)
        y = swoosh_l_forward(x_)
        y.backward(torch.ones_like(y))
    return y.detach(), x_.grad.detach()


def swoosh_r_forward_and_deriv(x):
    with torch.enable_grad():
        x_ = x.detach().requires_grad_(True)
        y = swoosh_r_forward(x_)
        y.backward(torch.ones_like(y))
    return y.detach(), x_.grad.detach()
