from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import onnxruntime as ort
except Exception:
    ort = None


def adain_style_transfer(content, style, alpha=1.0, eps=1e-6):
    """
    Lightweight AdaIN-style transfer in pixel space.
    Inputs are expected in [-1, 1], outputs in [-1, 1].
    """
    if not isinstance(content, torch.Tensor) or not isinstance(style, torch.Tensor):
        raise TypeError("content and style must be torch.Tensor.")
    if content.ndim != 4:
        raise ValueError(f"content must be [B,C,H,W], got shape={tuple(content.shape)}")
    if style.ndim == 3:
        style = style.unsqueeze(0)
    if style.ndim != 4:
        raise ValueError(f"style must be [1,C,H,W] or [B,C,H,W], got shape={tuple(style.shape)}")
    if content.shape[1] != style.shape[1]:
        raise ValueError(
            f"channel mismatch between content ({content.shape[1]}) and style ({style.shape[1]})."
        )

    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    out_dtype = content.dtype
    content_01 = (content / 2.0 + 0.5).clamp(0.0, 1.0).float()
    style_01 = (style / 2.0 + 0.5).clamp(0.0, 1.0).to(device=content.device, dtype=torch.float32)

    if style_01.shape[0] == 1 and content_01.shape[0] > 1:
        style_01 = style_01.repeat(content_01.shape[0], 1, 1, 1)
    elif style_01.shape[0] != content_01.shape[0]:
        raise ValueError(
            f"style batch size must be 1 or match content batch size, got {style_01.shape[0]} and {content_01.shape[0]}"
        )

    c_mean = content_01.mean(dim=(2, 3), keepdim=True)
    c_std = content_01.std(dim=(2, 3), keepdim=True, unbiased=False).clamp(min=eps)
    s_mean = style_01.mean(dim=(2, 3), keepdim=True)
    s_std = style_01.std(dim=(2, 3), keepdim=True, unbiased=False).clamp(min=eps)

    normalized = (content_01 - c_mean) / c_std
    stylized = normalized * s_std + s_mean
    mixed = alpha * stylized + (1.0 - alpha) * content_01
    mixed = mixed.clamp(0.0, 1.0)
    return (mixed * 2.0 - 1.0).clamp(-1.0, 1.0).to(dtype=out_dtype)


class AdaINStyleTransfer:
    def __init__(self, style_tensor, alpha=1.0):
        if not isinstance(style_tensor, torch.Tensor):
            raise TypeError("style_tensor must be torch.Tensor.")
        if style_tensor.ndim == 3:
            style_tensor = style_tensor.unsqueeze(0)
        if style_tensor.ndim != 4:
            raise ValueError(f"style_tensor must be [1,C,H,W] or [B,C,H,W], got {tuple(style_tensor.shape)}")
        self.style_tensor = style_tensor
        self.alpha = float(alpha)

    def __call__(self, content):
        style = self.style_tensor.to(device=content.device, dtype=content.dtype)
        return adain_style_transfer(content, style, alpha=self.alpha)


class ONNXMosaicStyleTransfer:
    """
    ONNX mosaic style transfer.
    Model input/output are expected in [0,255], shape [N,3,H,W].
    """

    def __init__(self, model_path, prefer_cuda=True):
        if ort is None:
            raise ImportError(
                "onnxruntime is not installed. "
                "Install it with: pip install onnxruntime"
            )

        model_path = Path(model_path).expanduser()
        if not model_path.is_absolute():
            model_path = model_path.resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX style model not found: {model_path}")

        available = set(ort.get_available_providers())
        providers = []
        if prefer_cuda and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        # Avoid affinity warnings in constrained CPU sets.
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        # Suppress verbose graph warnings from exported zoo models.
        sess_opts.log_severity_level = 3

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=providers if providers else None,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.expected_batch = self._dim_to_int(self.input_shape[0]) if len(self.input_shape) > 0 else None
        self.expected_h = self._dim_to_int(self.input_shape[2]) if len(self.input_shape) > 2 else None
        self.expected_w = self._dim_to_int(self.input_shape[3]) if len(self.input_shape) > 3 else None

    @staticmethod
    def _dim_to_int(dim):
        if isinstance(dim, int):
            return dim
        return None

    def __call__(self, content):
        if not isinstance(content, torch.Tensor) or content.ndim != 4:
            raise ValueError("content must be a torch.Tensor with shape [B,C,H,W].")

        out_dtype = content.dtype
        original_h, original_w = content.shape[2], content.shape[3]
        inp = (content / 2.0 + 0.5).clamp(0.0, 1.0).mul(255.0).to(dtype=torch.float32)

        resized = False
        if self.expected_h is not None and self.expected_w is not None:
            if inp.shape[2] != self.expected_h or inp.shape[3] != self.expected_w:
                inp = F.interpolate(inp, size=(self.expected_h, self.expected_w), mode="bilinear", align_corners=False)
                resized = True

        if self.expected_batch == 1 and inp.shape[0] != 1:
            out_chunks = []
            for i in range(inp.shape[0]):
                sub = inp[i : i + 1].detach().cpu().numpy()
                out_sub = self.session.run([self.output_name], {self.input_name: sub})[0]
                out_chunks.append(torch.from_numpy(out_sub))
            out = torch.cat(out_chunks, dim=0).to(device=content.device, dtype=torch.float32)
        else:
            inp_np = inp.detach().cpu().numpy()
            out_np = self.session.run([self.output_name], {self.input_name: inp_np})[0]
            out = torch.from_numpy(out_np).to(device=content.device, dtype=torch.float32)

        if resized:
            out = F.interpolate(out, size=(original_h, original_w), mode="bilinear", align_corners=False)
        out = out.clamp(0.0, 255.0).div(255.0)
        return (out * 2.0 - 1.0).clamp(-1.0, 1.0).to(dtype=out_dtype)
