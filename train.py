# train_keyalign.py
import argparse
from typing import Dict, Callable, Tuple

import torch
from ultralytics import YOLO


# -------------------------
# YOLOv10 E2E sanity checks
# -------------------------
def _parse_torch_device(device_str: str) -> torch.device:
    """Best-effort parse of Ultralytics --device strings ('0', '0,1', 'cuda:0', 'cpu')."""
    s = (device_str or "").strip().lower()
    if s in {"cpu", "-1", "none"}:
        return torch.device("cpu")

    # If user passed "0,1" etc., take the first GPU id.
    first = s.split(",")[0].strip()
    if first.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{first}")
        return torch.device("cpu")

    # "cuda:0" etc.
    if first.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(first)
        return torch.device("cpu")

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _get_last_head_module(yolo: YOLO):
    """Return (net, last_module, last_module_name) robustly across Ultralytics versions."""
    net = getattr(yolo, "model", None)
    if net is None:
        return None, None, None

    mods = getattr(net, "model", None)
    last = None
    if isinstance(mods, (list, tuple)):
        last = mods[-1] if mods else None
    else:
        # nn.Sequential / nn.ModuleList support indexing
        try:
            last = mods[-1]  # type: ignore[index]
        except Exception:
            last = None

    return (
        net,
        last,
        (
            getattr(last, "__class__", type("X", (), {})).__name__
            if last is not None
            else None
        ),
    )


def sanity_check_yolov10_e2e(
    yolo: YOLO, imgsz: int, device_str: str, verbose: bool = True
) -> bool:
    """
    Check whether a YOLOv10-style end-to-end (E2E) model is wired correctly:
      - last head is v10Detect-ish
      - model/end2end flags are True when expected
      - init_criterion() resolves to E2EDetectLoss when end2end=True
      - forward() output contains 'one2many'/'one2one' dict keys when end2end=True
      - stride tensor exists and matches number of detection layers
    Returns True if checks look OK, False if something looks off.
    """
    ok = True
    net, last, last_name = _get_last_head_module(yolo)

    if net is None:
        print("[SanityCheck] Could not access yolo.model (unexpected).")
        return False

    end2end_model = bool(getattr(net, "end2end", False))
    end2end_head = bool(getattr(last, "end2end", False)) if last is not None else False

    if verbose:
        print("\n[SanityCheck] YOLOv10 E2E wiring")
        print(f"  last_head:            {last_name}")
        print(f"  net.end2end:          {end2end_model}")
        print(f"  head.end2end:         {end2end_head}")
        print(f"  net.stride:           {getattr(net, 'stride', None)}")

    # Heuristic: v10 configs should end with v10Detect
    if last_name is None or "v10" not in last_name.lower():
        ok = False
        print(
            "  [WARN] Head doesn't look like v10Detect. If you're expecting YOLOv10 E2E, this is suspicious."
        )

    # Criterion selection in tasks.py:
    # DetectionModel.init_criterion returns E2EDetectLoss iff getattr(self,'end2end',False) is True.
    # (see tasks.py init_criterion) fileciteturn16file0L69-L75
    try:
        crit = net.init_criterion()  # creates E2EDetectLoss or v8DetectionLoss
        crit_name = crit.__class__.__name__
        if verbose:
            print(f"  init_criterion():      {crit_name}")
        if end2end_model and "E2E" not in crit_name and "E2e" not in crit_name:
            ok = False
            print(
                "  [WARN] net.end2end=True but init_criterion() is NOT E2EDetectLoss (training may behave like v8)."
            )
        if (not end2end_model) and ("E2E" in crit_name or "E2e" in crit_name):
            ok = False
            print(
                "  [WARN] net.end2end=False but init_criterion() returned an E2E loss (unexpected)."
            )
    except Exception as e:
        ok = False
        import traceback

        print(f"  [WARN] init_criterion() failed: {type(e).__name__}: {e}")
        # Print a short traceback to pinpoint the exact file/line in your environment.
        tb = traceback.format_exc(limit=6)
        for ln in tb.strip().splitlines():
            print(f"    {ln}")

        # Helpful context for the common alpha_iou UnboundLocalError.
        bbox_loss_type = getattr(net, "bbox_loss_type", None)
        alpha_iou_attr = getattr(net, "alpha_iou", None)
        print(f"  [Context] net.bbox_loss_type:  {bbox_loss_type}")
        print(f"  [Context] net.alpha_iou:       {alpha_iou_attr}")

        if "alpha_iou" in str(e).lower():
            print(
                "  [HINT] This usually means E2EDetectLoss references local variable `alpha_iou`"
                " before assignment. Fix by defining `alpha_iou = getattr(model, 'alpha_iou', 3.0)`"
                " unconditionally, and only using the exponent when bbox_loss_type=='alpha_iou'."
            )

    # Forward-output check for dict keys used by stride builder when end2end=True:
    # In DetectionModel stride build, if self.end2end it uses self.forward(x)['one2many'] fileciteturn16file3L5-L20
    dev = _parse_torch_device(device_str)
    try:
        net = net.to(dev)
        x = torch.zeros(1, 3, int(imgsz), int(imgsz), device=dev)

        # Try train-mode forward first (some heads only emit dict keys in training mode)
        net.train()
        with torch.no_grad():
            out = net(x)

        # Fallback to eval-mode if needed
        if out is None:
            net.eval()
            with torch.no_grad():
                out = net(x)

        if verbose:
            print(f"  forward_out_type:      {type(out).__name__}")

        out_dict = None
        if isinstance(out, dict):
            out_dict = out
        elif isinstance(out, (list, tuple)) and out and isinstance(out[0], dict):
            out_dict = out[0]

        if end2end_model:
            if out_dict is None:
                ok = False
                print(
                    "  [WARN] end2end=True but forward() did not return a dict. Expected keys: 'one2many'/'one2one'."
                )
            else:
                keys = set(out_dict.keys())
                if verbose:
                    print(f"  forward_out_keys:      {sorted(list(keys))}")
                if "one2many" not in keys:
                    ok = False
                    print(
                        "  [WARN] Missing 'one2many' in forward() output dict (stride builder + loss may break)."
                    )
                if "one2one" not in keys:
                    # some forks may omit one2one, but YOLOv10 E2E typically has both
                    print(
                        "  [WARN] Missing 'one2one' in forward() output dict (check your v10Detect forward)."
                    )

        # Stride sanity: should be a tensor/list with length = nl (num detection layers)
        stride = getattr(net, "stride", None)
        nl = getattr(last, "nl", None) if last is not None else None
        if stride is None:
            print("  [WARN] net.stride is None (stride build may have failed).")
        elif nl is not None:
            try:
                stride_len = len(stride)
                if stride_len != int(nl):
                    ok = False
                    print(
                        f"  [WARN] stride length ({stride_len}) != head.nl ({nl}). Check head outputs/P-levels."
                    )
            except Exception:
                # if stride is scalar tensor
                pass

    except Exception as e:
        ok = False
        print(f"  [WARN] forward() sanity run failed: {type(e).__name__}: {e}")

    if verbose:
        print(f"  result:               {'OK' if ok else 'CHECK FAILED'}\n")
    return ok


def parse_args():
    p = argparse.ArgumentParser(
        description="Train/finetune/resume YOLO (Ultralytics). YAML defines architecture; --weights loads partial state_dict (no arch override)."
    )

    # Model / weights
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="*.yaml/*.yml (build architecture) OR *.pt (train that checkpoint architecture).",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional *.pt pretrained weights to load INTO a YAML-defined model (partial load by key+shape).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a checkpoint (*.pt). Use with --model last.pt or --weights last.pt.",
    )

    # Custom loss args (your fork reads these)
    p.add_argument(
        "--bbox_loss_type",
        type=str,
        default="ciou",
        choices=["iou", "giou", "diou", "ciou", "eiou", "siou", "alpha_iou"],
        help="IoU variant used in bbox regression loss.",
    )
    p.add_argument(
        "--alpha_iou",
        type=float,
        default=2.0,
        help="Exponent for alpha-IoU (used only when bbox_loss_type='alpha_iou').",
    )

    # CALM-Row navigation-native loss (optional, no new params; see ultralytics/utils/calm_row.py)
    p.add_argument(
        "--calm_row",
        action="store_true",
        help="Enable CALM-Row uncertainty-weighted guidance-line loss (DFL-variance based).",
    )
    p.add_argument("--calm_calib_gain", type=float, default=0.5,
                   help="Gain for the CALM-Row DDC calibration (NLL) loss.")
    p.add_argument("--calm_line_gain", type=float, default=1.0,
                   help="Gain for the CALM-Row line-stability loss.")
    p.add_argument("--calm_y_far", type=float, default=0.0,
                   help="Look-ahead row (px) the line loss extrapolates to (0 = image top).")
    p.add_argument("--calm_lam_theta", type=float, default=0.5,
                   help="Weight of the heading term vs RMSE in the CALM-Row line-stability loss.")

    # Training args
    p.add_argument("--data", type=str, required=True, help="data.yaml path")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", type=str, default="0")
    p.add_argument(
        "--dir",
        type=str,
        default="runs/detect",
        help="Root directory to save results (e.g., /content/my_runs or runs/detect)",
    )
    p.add_argument("--name", type=str, default="exp")
    p.add_argument(
        "--prune", action="store_true", help="Enable pruning (your fork-specific flag)."
    )

    # Debug
    p.add_argument(
        "--debug_keys",
        action="store_true",
        help="Print sample keys from checkpoint and model (helps diagnose 0% load).",
    )

    p.add_argument(
        "--skip_sanity",
        action="store_true",
        help="Skip YOLOv10 E2E wiring sanity checks (recommended to keep ON while iterating heads).",
    )

    return p.parse_args()


def _is_yaml(x: str) -> bool:
    x = (x or "").lower()
    return x.endswith(".yaml") or x.endswith(".yml")


def _is_pt(x: str) -> bool:
    x = (x or "").lower()
    return x.endswith(".pt")


def _extract_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Extract a usable state_dict mapping from an Ultralytics *.pt checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = None
    if isinstance(ckpt, dict):
        if "ema" in ckpt and hasattr(ckpt["ema"], "state_dict"):
            state = ckpt["ema"].state_dict()
        elif "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
            state = ckpt["model"].state_dict()
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            tensor_items = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
            if tensor_items:
                state = tensor_items
    elif hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()

    if state is None or not isinstance(state, dict) or not state:
        raise ValueError(f"Could not extract state_dict from checkpoint: {ckpt_path}")

    return state


def _make_key_transforms(prefixes):
    def identity(k: str) -> str:
        return k

    def strip_once(pre: str):
        def f(k: str) -> str:
            return k[len(pre) :] if k.startswith(pre) else k

        return f

    def strip_all(pre: str):
        def f(k: str) -> str:
            while k.startswith(pre):
                k = k[len(pre) :]
            return k

        return f

    def add_prefix(pre: str):
        def f(k: str) -> str:
            return pre + k

        return f

    transforms: Dict[str, Callable[[str], str]] = {"identity": identity}
    for pre in prefixes:
        transforms[f"strip_once:{pre}"] = strip_once(pre)
        transforms[f"strip_all:{pre}"] = strip_all(pre)
        transforms[f"add_prefix:{pre}"] = add_prefix(pre)
    return transforms


def _best_align_state_keys(
    state: Dict[str, torch.Tensor], target_keys: set
) -> Tuple[Dict[str, torch.Tensor], str, int]:
    """Choose a key transform that maximizes intersection with target keys."""
    prefixes = ["model.model.", "model.", "module.", "ema."]
    transforms = _make_key_transforms(prefixes)

    best_name = "identity"
    best_match = -1
    best_state = state

    for name, tf in transforms.items():
        mapped_keys = [tf(k) for k in state.keys()]
        match = sum(1 for k in mapped_keys if k in target_keys)
        if match > best_match:
            best_match = match
            best_name = name
            best_state = {tf(k): v for k, v in state.items()}

    return best_state, best_name, best_match


def load_weights_into_yaml_model(
    yolo: YOLO, weights_path: str, debug_keys: bool = False
) -> None:
    """Partial-load weights into a YAML-built model by matching keys + tensor shapes."""
    state_raw = _extract_state_dict(weights_path)
    target_state = yolo.model.state_dict()
    target_keys = set(target_state.keys())

    # Align key naming (prevents 0% load caused by prefix mismatch like 'model.' vs no prefix)
    state, align_name, align_hits = _best_align_state_keys(state_raw, target_keys)

    filtered = {}
    skipped_missing = 0
    skipped_shape = 0

    for k, v in state.items():
        if k not in target_state:
            skipped_missing += 1
            continue
        if tuple(v.shape) != tuple(target_state[k].shape):
            skipped_shape += 1
            continue
        filtered[k] = v

    incompatible = yolo.model.load_state_dict(filtered, strict=False)

    loaded_params = sum(t.numel() for t in filtered.values())
    total_params = sum(t.numel() for t in target_state.values())
    ratio = loaded_params / max(total_params, 1)

    miss = getattr(incompatible, "missing_keys", [])
    unexp = getattr(incompatible, "unexpected_keys", [])

    print("\n[Weights Load Report]")
    print(f"  weights_path:        {weights_path}")
    print(f"  key_align:           {align_name} (initial key hits={align_hits})")
    print(f"  YAML model params:   {total_params:,}")
    print(f"  Loaded params:       {loaded_params:,} ({ratio*100:.2f}%)")
    print(f"  Keys loaded:         {len(filtered):,}")
    print(f"  Skipped (missing):   {skipped_missing:,}")
    print(f"  Skipped (shape):     {skipped_shape:,}")
    print(f"  Missing keys (after load):   {len(miss):,}")
    print(f"  Unexpected keys (after load):{len(unexp):,}\n")

    if debug_keys:
        ck_keys = list(state_raw.keys())
        tg_keys = list(target_state.keys())
        print("[Debug Keys] checkpoint sample:")
        for k in ck_keys[:10]:
            print(" ", k)
        print("[Debug Keys] model sample:")
        for k in tg_keys[:10]:
            print(" ", k)
        print("")


def build_model(args) -> YOLO:
    if args.resume:
        ckpt = args.weights if args.weights else args.model
        if not ckpt or not _is_pt(ckpt):
            raise ValueError(
                "--resume requires a *.pt checkpoint via --model last.pt or --weights last.pt"
            )
        print(f"[Build] Resume from checkpoint: {ckpt}")
        return YOLO(ckpt)

    if _is_yaml(args.model):
        print(f"[Build] YAML architecture: {args.model}")
        m = YOLO(args.model)

        if args.weights:
            if not _is_pt(args.weights):
                raise ValueError("--weights must be a *.pt file")
            print(
                f"[Build] Loading weights into YAML (no arch override): {args.weights}"
            )
            load_weights_into_yaml_model(m, args.weights, debug_keys=args.debug_keys)

        return m

    if _is_pt(args.model):
        if args.weights:
            raise ValueError(
                "When --model is *.pt, do not pass --weights. Use --resume to resume that checkpoint."
            )
        print(f"[Build] PT model (architecture from checkpoint): {args.model}")
        return YOLO(args.model)

    raise ValueError("--model must be a *.yaml/*.yml or *.pt")


def main():
    args = parse_args()
    model = build_model(args)

    if not args.resume:
        setattr(model.model, "bbox_loss_type", args.bbox_loss_type)
        setattr(model.model, "alpha_iou", args.alpha_iou)
        setattr(model.model, "calm_row", args.calm_row)
        setattr(model.model, "calm_calib_gain", args.calm_calib_gain)
        setattr(model.model, "calm_line_gain", args.calm_line_gain)
        setattr(model.model, "calm_y_far", args.calm_y_far)
        setattr(model.model, "calm_lam_theta", args.calm_lam_theta)

    print("[Debug] cfg:", getattr(model, "cfg", None))
    print("[Debug] ckpt_path:", getattr(model, "ckpt_path", None))
    try:
        model.info()
    except Exception:
        try:
            model.model.info()
        except Exception:
            pass

    # Sanity-check YOLOv10 E2E wiring (helps catch 'train v10 like v8' issues)
    if not args.skip_sanity:
        sanity_check_yolov10_e2e(
            model, imgsz=args.imgsz, device_str=args.device, verbose=True
        )

    if args.resume:
        results = model.train(resume=True, prune=False, prune_load=False)
    else:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.dir,
            name=args.name,
            prune=args.prune,
            workers=0,
            cache=False,
            verbose=True,
        )

    print("Training finished.")
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
