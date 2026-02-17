import math

import lpips as LPIPS
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch import optim
from tqdm import tqdm

def _build_lpips_model(device):
    lpips_model = LPIPS.LPIPS(net="alex").to(device).eval()
    for param in lpips_model.parameters():
        param.requires_grad_(False)
    return lpips_model


def _lpips_per_sample(lpips_model, x1, x2):
    # LPIPS expects float tensor in [-1, 1], output shape is [B, 1, 1, 1].
    return lpips_model(x1.float(), x2.float()).flatten()


def _flatten_l2_per_sample(tensor):
    return tensor.float().reshape(tensor.shape[0], -1).norm(p=2, dim=1)


def _normalized_l2_per_sample(delta):
    d = float(delta[0].numel())
    return _flatten_l2_per_sample(delta) / math.sqrt(d)


def _linf_per_sample(delta):
    return delta.reshape(delta.shape[0], -1).abs().amax(dim=1)


def _project_l2_normed(delta, budget_l2_normed):
    if budget_l2_normed is None:
        return delta
    flat = delta.reshape(delta.shape[0], -1)
    l2_norm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    max_l2 = budget_l2_normed * math.sqrt(float(delta[0].numel()))
    scale = torch.clamp(max_l2 / l2_norm, max=1.0)
    return (flat * scale).reshape_as(delta)


def _project_linf(delta, eps):
    if eps is None:
        return delta
    return delta.clamp(-eps, eps)


def _enforce_lpips_budget(X, X_adv, lpips_model, budget_lpips, bisection_steps=6):
    if budget_lpips is None:
        return X_adv

    with torch.no_grad():
        current_lpips = _lpips_per_sample(lpips_model, X, X_adv)
        over_budget_mask = current_lpips > budget_lpips
        if not torch.any(over_budget_mask):
            return X_adv

        X_adv_new = X_adv.clone()
        X_ref = X[over_budget_mask]
        delta_ref = X_adv[over_budget_mask] - X_ref
        lo = torch.zeros(X_ref.shape[0], device=X.device, dtype=X.dtype)
        hi = torch.ones(X_ref.shape[0], device=X.device, dtype=X.dtype)

        for _ in range(max(1, int(bisection_steps))):
            mid = (lo + hi) / 2
            x_mid = (X_ref + mid.view(-1, 1, 1, 1) * delta_ref).clamp(-1, 1)
            lp_mid = _lpips_per_sample(lpips_model, X_ref, x_mid)
            too_high = lp_mid > budget_lpips
            hi = torch.where(too_high, mid, hi)
            lo = torch.where(too_high, lo, mid)

        alpha = lo.view(-1, 1, 1, 1)
        X_adv_new[over_budget_mask] = (X_ref + alpha * delta_ref).clamp(-1, 1)
    return X_adv_new


def _prepare_target_tensor(target_image, X):
    if target_image is None:
        return None
    if isinstance(target_image, torch.Tensor):
        target = target_image.to(device=X.device, dtype=X.dtype)
    else:
        raise TypeError("target_image must be a torch.Tensor in [-1, 1].")
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if target.shape[0] == 1 and X.shape[0] > 1:
        target = target.repeat(X.shape[0], 1, 1, 1)
    if target.shape != X.shape:
        raise ValueError(f"target_image shape mismatch, expected {tuple(X.shape)}, got {tuple(target.shape)}")
    return target.clamp(-1, 1)


def _attack_objective(
    method,
    X,
    latent_X,
    decoded_X_adv,
    latent_X_adv,
    lpips_model,
    latent_target=None,
    target_tensor=None,
):
    method = method.lower()
    if method == "chocolatent":
        if decoded_X_adv is None:
            raise ValueError("method=chocolatent requires decoded_X_adv tensor.")
        attack_score = _lpips_per_sample(lpips_model, X, decoded_X_adv)
    elif method in {"robust-ldm", "robust_ldm"}:
        attack_score = _flatten_l2_per_sample(latent_X_adv - latent_X)
    elif method in {"glaze", "photoguard", "mist"}:
        if latent_target is None:
            if method == "photoguard":
                raise ValueError("method=photoguard requires latent target tensor.")
            if method == "mist":
                raise ValueError("method=mist requires target_image tensor in [-1, 1].")
            raise ValueError("method=glaze requires style target image tensor in [-1, 1].")
        attack_score = -_flatten_l2_per_sample(latent_X_adv - latent_target)
    else:
        raise ValueError(f"Unknown method: {method}")
    return attack_score


def _collect_final_metrics(
    X,
    X_adv,
    latent_X,
    latent_X_adv,
    decoded_X_adv,
    lpips_model,
    attack_score,
):
    with torch.no_grad():
        X_adv_safe = torch.nan_to_num(X_adv, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
        decoded_X_adv_safe = torch.nan_to_num(decoded_X_adv, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
        latent_X_adv_safe = torch.nan_to_num(latent_X_adv, nan=0.0, posinf=0.0, neginf=0.0)
        attack_score_safe = torch.nan_to_num(attack_score.float(), nan=0.0, posinf=0.0, neginf=0.0)

        delta = X_adv_safe - X
        input_l2 = _flatten_l2_per_sample(delta)
        input_l2_normed = _normalized_l2_per_sample(delta)
        input_linf = _linf_per_sample(delta)
        input_lpips = _lpips_per_sample(lpips_model, X, X_adv_safe)
        decoded_lpips = _lpips_per_sample(lpips_model, X, decoded_X_adv_safe)
        decoded_l2 = _flatten_l2_per_sample(decoded_X_adv_safe - X)
        latent_l2 = _flatten_l2_per_sample(latent_X_adv_safe - latent_X)

        x_np = X.detach().float().cpu().numpy()
        x_adv_np = X_adv_safe.detach().float().cpu().numpy()
        x_dec_np = decoded_X_adv_safe.detach().float().cpu().numpy()

        input_psnr = []
        input_ssim = []
        decoded_psnr = []
        decoded_ssim = []
        for x_i, x_adv_i, x_dec_i in zip(x_np, x_adv_np, x_dec_np):
            input_psnr.append(float(psnr(x_i, x_adv_i, data_range=2.0)))
            input_ssim.append(float(ssim(x_i, x_adv_i, data_range=2.0, channel_axis=0)))
            decoded_psnr.append(float(psnr(x_i, x_dec_i, data_range=2.0)))
            decoded_ssim.append(float(ssim(x_i, x_dec_i, data_range=2.0, channel_axis=0)))

    return {
        "attack_score": attack_score_safe.detach().cpu().tolist(),
        "input_l2": input_l2.detach().float().cpu().tolist(),
        "input_l2_normed": input_l2_normed.detach().float().cpu().tolist(),
        "input_linf": input_linf.detach().float().cpu().tolist(),
        "input_lpips": input_lpips.detach().float().cpu().tolist(),
        "decoded_lpips": decoded_lpips.detach().float().cpu().tolist(),
        "decoded_l2": decoded_l2.detach().float().cpu().tolist(),
        "latent_l2": latent_l2.detach().float().cpu().tolist(),
        "input_psnr": input_psnr,
        "input_ssim": input_ssim,
        "decoded_psnr": decoded_psnr,
        "decoded_ssim": decoded_ssim,
    }


def _all_finite(tensor):
    return bool(torch.isfinite(tensor).all())


def _infer_model_dtype(model, fallback_dtype):
    if hasattr(model, "parameters"):
        for param in model.parameters():
            return param.dtype
    return fallback_dtype


def _make_feasible_adv(
    X,
    budget_l2_normed,
    budget_lpips,
    eps,
    lpips_model,
    strict_lpips_projection,
    lpips_bisection_steps,
):
    delta = torch.empty_like(X).uniform_(-1, 1)
    delta = _project_l2_normed(delta, budget_l2_normed)
    delta = _project_linf(delta, eps)
    X_adv = (X + delta).clamp(-1, 1)
    if strict_lpips_projection:
        X_adv = _enforce_lpips_budget(
            X, X_adv, lpips_model, budget_lpips=budget_lpips, bisection_steps=lpips_bisection_steps
        )
    return X_adv


def pgd(
    X,
    model,
    iters=200,
    max_img_Dis=15,
    max_img_lpips=0.15,
    initial_lr=0.001,
    eps=0.1,
    **kwargs,
):
    show_progress = kwargs.get("show_progress", True)
    log_every = max(1, int(kwargs.get("log_every", max(iters // 4, 1))))
    step_collector = kwargs.get("step_collector")

    method = kwargs.get("method", "chocolatent").lower()
    strict_lpips_projection = kwargs.get("strict_lpips_projection", True)
    lpips_bisection_steps = kwargs.get("lpips_bisection_steps", 6)
    budget_penalty_l2 = float(kwargs.get("budget_penalty_l2", 20.0))
    budget_penalty_lpips = float(kwargs.get("budget_penalty_lpips", 20.0))
    nan_lr_decay = float(kwargs.get("nan_lr_decay", 0.5))
    nan_min_lr = float(kwargs.get("nan_min_lr", 1e-4))
    nan_max_recoveries = int(kwargs.get("nan_max_recoveries", 8))

    nan_lr_decay = min(max(nan_lr_decay, 0.1), 0.95)
    nan_min_lr = max(nan_min_lr, 1e-8)
    nan_max_recoveries = max(1, nan_max_recoveries)

    X = torch.nan_to_num(X.detach().float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
    d_sqrt = math.sqrt(float(X[0].numel()))
    budget_l2_normed = kwargs.get("budget_l2_normed")
    if budget_l2_normed is None:
        raw_budget_l2 = kwargs.get("budget_l2")
        if raw_budget_l2 is None:
            # Backward compatibility: old max_img_Dis was absolute L2.
            raw_budget_l2 = float(max_img_Dis) / d_sqrt
        budget_l2_normed = float(raw_budget_l2)

    budget_lpips = kwargs.get("budget_lpips")
    if budget_lpips is None:
        budget_lpips = float(max_img_lpips)

    if eps is not None and eps <= 0:
        eps = None

    lpips_model = kwargs.get("lpips_model")
    if lpips_model is None:
        lpips_model = _build_lpips_model(X.device)

    target_tensor = _prepare_target_tensor(kwargs.get("target_image"), X)
    return_info = kwargs.get("return_info", False)
    model_forward_dtype = _infer_model_dtype(model, X.dtype)

    def encode_model_input(tensor):
        return model.encode(tensor.to(dtype=model_forward_dtype)).latent_dist.mean

    def decode_model_latent(latent):
        return model.decode(latent.to(dtype=model_forward_dtype)).sample

    if show_progress:
        print(
            f"method={method}, X.shape={tuple(X.shape)}, iters={iters}, "
            f"budget_l2_normed={budget_l2_normed:.6f}, budget_lpips={budget_lpips:.6f}, eps={eps}"
        )

    X_adv = _make_feasible_adv(
        X,
        budget_l2_normed=budget_l2_normed,
        budget_lpips=budget_lpips,
        eps=eps,
        lpips_model=lpips_model,
        strict_lpips_projection=strict_lpips_projection,
        lpips_bisection_steps=lpips_bisection_steps,
    ).detach()
    X_adv.requires_grad_(True)

    with torch.no_grad():
        latent_X = encode_model_input(X).float()
        latent_target = None
        if method in {"glaze", "photoguard", "mist"}:
            if method == "photoguard":
                # PhotoGuard target is an all-zero tensor in latent space.
                latent_target = torch.zeros_like(latent_X)
            else:
                if target_tensor is None:
                    if method == "mist":
                        raise ValueError("method=mist requires target_image tensor in [-1, 1].")
                    raise ValueError("method=glaze requires style target image tensor in [-1, 1].")
                latent_target = encode_model_input(target_tensor).float()

    current_lr = max(float(initial_lr), nan_min_lr)
    optimizer = optim.Adam([X_adv], lr=current_lr)
    iterator = tqdm(range(iters), disable=not show_progress)
    best_objective = float("-inf")
    best_x_adv = X_adv.detach().clone()
    nan_recoveries = 0

    def recover_from_non_finite(step_idx: int, reason: str):
        nonlocal X_adv, optimizer, current_lr, nan_recoveries, best_x_adv

        nan_recoveries += 1
        current_lr = max(nan_min_lr, current_lr * nan_lr_decay)
        if best_x_adv is not None and _all_finite(best_x_adv):
            recovered = best_x_adv.detach().clone()
        else:
            recovered = _make_feasible_adv(
                X,
                budget_l2_normed=budget_l2_normed,
                budget_lpips=budget_lpips,
                eps=eps,
                lpips_model=lpips_model,
                strict_lpips_projection=strict_lpips_projection,
                lpips_bisection_steps=lpips_bisection_steps,
            ).detach()

        X_adv = recovered.requires_grad_(True)
        optimizer = optim.Adam([X_adv], lr=current_lr)

        if show_progress:
            print(
                f"[Warn] Non-finite detected at iter={step_idx} ({reason}). "
                f"recovery={nan_recoveries}/{nan_max_recoveries}, lr={current_lr:.6g}",
                flush=True,
            )

        return nan_recoveries <= nan_max_recoveries

    for i in iterator:
        latent_X_adv = encode_model_input(X_adv)
        latent_X_adv = latent_X_adv.float()
        decoded_X_adv = None
        if method == "chocolatent":
            decoded_X_adv = decode_model_latent(latent_X_adv).float()
            if not _all_finite(latent_X_adv) or not _all_finite(decoded_X_adv):
                if not recover_from_non_finite(i, "latent/decode"):
                    break
                continue
        elif not _all_finite(latent_X_adv):
            if not recover_from_non_finite(i, "latent"):
                break
            continue

        attack_score = _attack_objective(
            method=method,
            X=X,
            latent_X=latent_X,
            decoded_X_adv=decoded_X_adv,
            latent_X_adv=latent_X_adv,
            lpips_model=lpips_model,
            latent_target=latent_target,
            target_tensor=target_tensor,
        )
        if not _all_finite(attack_score):
            if not recover_from_non_finite(i, "attack_score"):
                break
            continue
        score_mean = attack_score.mean()

        delta = X_adv - X
        l2_normed = _normalized_l2_per_sample(delta)
        input_lpips = _lpips_per_sample(lpips_model, X, X_adv)
        if not _all_finite(l2_normed) or not _all_finite(input_lpips):
            if not recover_from_non_finite(i, "constraints"):
                break
            continue
        l2_penalty = torch.relu(l2_normed - budget_l2_normed).mean()
        lpips_penalty = torch.relu(input_lpips - budget_lpips).mean()
        budget_penalty = budget_penalty_l2 * l2_penalty + budget_penalty_lpips * lpips_penalty

        objective = score_mean - budget_penalty
        loss = -objective
        if not _all_finite(objective) or not _all_finite(loss):
            if not recover_from_non_finite(i, "objective/loss"):
                break
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if X_adv.grad is None or (not _all_finite(X_adv.grad)):
            if not recover_from_non_finite(i, "grad"):
                break
            continue
        optimizer.step()

        with torch.no_grad():
            delta = X_adv - X
            delta = _project_linf(delta, eps)
            delta = _project_l2_normed(delta, budget_l2_normed)
            X_adv.data.copy_((X + delta).clamp(-1, 1))
            if strict_lpips_projection:
                X_adv.data.copy_(
                    _enforce_lpips_budget(
                        X,
                        X_adv.data,
                        lpips_model,
                        budget_lpips=budget_lpips,
                        bisection_steps=lpips_bisection_steps,
                    )
                )
            if not _all_finite(X_adv):
                if not recover_from_non_finite(i, "post-projection"):
                    break
                continue

        objective_value = float(objective.detach().item())
        if math.isfinite(objective_value) and objective_value > best_objective:
            best_objective = objective_value
            best_x_adv = X_adv.detach().clone()

        should_log = (step_collector is not None or show_progress) and ((i % log_every == 0) or (i == iters - 1))
        if not should_log:
            continue

        with torch.no_grad():
            if decoded_X_adv is None:
                decoded_for_log = decode_model_latent(latent_X_adv.detach()).float()
            else:
                decoded_for_log = decoded_X_adv
            decoded_lpips = _lpips_per_sample(lpips_model, X, decoded_for_log).mean().item()
            metrics = {
                "attack_score": score_mean.detach().item(),
                "objective": objective.detach().item(),
                "budget_penalty": budget_penalty.detach().item(),
                "input_l2_normed": l2_normed.mean().detach().item(),
                "input_lpips": input_lpips.mean().detach().item(),
                "decoded_lpips": decoded_lpips,
            }

        if step_collector is not None:
            step_collector.record_step(i, metrics)

        if show_progress:
            iterator.set_description(f"[{method}] objective={metrics['objective']:.4f}")
            print(
                " | ".join(
                    [
                        f"attack_score:{metrics['attack_score']:.4f}",
                        f"input_l2n:{metrics['input_l2_normed']:.4f}",
                        f"input_lpips:{metrics['input_lpips']:.4f}",
                        f"decoded_lpips:{metrics['decoded_lpips']:.4f}",
                    ]
                ),
                flush=True,
            )

    if not _all_finite(X_adv):
        X_adv = best_x_adv.detach().clone() if best_x_adv is not None else X.detach().clone()
    else:
        X_adv = torch.nan_to_num(X_adv.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

    with torch.no_grad():
        latent_X_adv = encode_model_input(X_adv).float()
        decoded_X_adv = decode_model_latent(latent_X_adv).float()
        final_attack_score = _attack_objective(
            method=method,
            X=X,
            latent_X=latent_X,
            decoded_X_adv=decoded_X_adv,
            latent_X_adv=latent_X_adv,
            lpips_model=lpips_model,
            latent_target=latent_target,
            target_tensor=target_tensor,
        )
        final_metrics = _collect_final_metrics(
            X=X,
            X_adv=X_adv,
            latent_X=latent_X,
            latent_X_adv=latent_X_adv,
            decoded_X_adv=decoded_X_adv,
            lpips_model=lpips_model,
            attack_score=final_attack_score,
        )

    if show_progress and X.shape[0] > 0:
        print(
            f"first_img input_psnr:{final_metrics['input_psnr'][0]:.3f} | "
            f"input_ssim:{final_metrics['input_ssim'][0]:.3f} | "
            f"decoded_psnr:{final_metrics['decoded_psnr'][0]:.3f} | "
            f"decoded_ssim:{final_metrics['decoded_ssim'][0]:.3f}",
            flush=True,
        )

    if return_info:
        return X_adv.detach(), {
            "method": method,
            "budget_l2_normed": float(budget_l2_normed),
            "budget_lpips": float(budget_lpips),
            "final_per_sample": final_metrics,
        }

    return X_adv.detach()
