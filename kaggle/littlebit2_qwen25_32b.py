#!/usr/bin/env python3
"""Streaming LittleBit-2 initializer for Qwen2.5-32B-Instruct on Kaggle T4 x2.

IMPORTANT: This is an initialization-only compression pass. It runs the official
SamsungLabs LittleBitLinear SVD + Joint-ITQ initializer, packs binary factors,
and preserves non-linear weights (including embeddings/lm_head) in their source
precision. It DOES NOT perform teacher/student QAT or claim lossless compression.

The source checkpoint is downloaded one safetensors shard at a time, quantized
on one of the two GPUs in parallel workers, and deleted after conversion. This
avoids ever materializing the full 32B BF16 model in host RAM or GPU memory.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

MODEL_DEFAULT = "Qwen/Qwen2.5-32B-Instruct"
UPSTREAM_REPO = "https://github.com/SamsungLabs/LittleBit.git"
UPSTREAM_COMMIT = "42d658b0c79f76450b34b6a3547462c7cdc3e1a0"


def log(message: str) -> None:
    print(f"[Sarvagya/LittleBit-2] {message}", flush=True)


def run(cmd: list[str], *, cwd: str | None = None) -> None:
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_dependencies() -> None:
    """Install only the Python packages this streaming converter needs.

    Keep the Kaggle-provided PyTorch/CUDA build intact; T4 does not support BF16
    tensor-core execution, and this script deliberately performs SVD in FP32.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Kaggle's preinstalled PyTorch is required; do not replace its CUDA build.") from exc

    # Install the pinned Transformers release without resolving its old
    # huggingface-hub<1 dependency constraint. Kaggle images may ship Hub 1.x
    # for Gradio/Diffusers; downgrading it creates unrelated environment conflicts.
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "transformers==4.51.3"])
    requirements = [
        "accelerate>=1.6.0",
        "safetensors>=0.4.5",
        "sentencepiece",
    ]
    run([sys.executable, "-m", "pip", "install", "-q", *requirements])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=MODEL_DEFAULT)
    p.add_argument("--output-dir", default=None,
                   help="Defaults to /kaggle/working/littlebit2-qwen2.5-32b-<eff-bit>bpw.")
    p.add_argument("--scratch-dir", default="/kaggle/temp/sarvagya_littlebit2")
    p.add_argument("--eff-bit", type=float, default=0.45,
                   help="Target BPW for eligible nn.Linear weights, not the whole model.")
    p.add_argument("--kv-factor", type=float, default=1.15,
                   help="Slightly increase rank for k_proj/v_proj for quality headroom.")
    p.add_argument("--itq-n-iter", type=int, default=10,
                   help="Use 10 for the 1-2h target profile; 50 is slower and closer to the paper-style initialization.")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--keep-source-shards", action="store_true",
                   help="Keep downloaded source shards in scratch (uses much more disk).")
    p.add_argument("--no-residual", action="store_true",
                   help="Disable the second residual factor path. Smaller/faster, usually less faithful.")
    p.add_argument("--skip-dependency-install", action="store_true")
    return p.parse_args()


def _free_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024 ** 3)


def resource_audit(scratch: Path, output: Path, workers: int) -> list[int]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. In Kaggle, set Accelerator = GPU T4 x2.")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(f"Expected two GPUs (T4 x2); detected {gpu_count}. Refusing to silently underutilize hardware.")

    gpu_ids = list(range(min(gpu_count, workers, 2)))
    log(f"Detected {gpu_count} CUDA GPUs")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        free_b, total_b = torch.cuda.mem_get_info(i)
        log(f"GPU {i}: {props.name}, VRAM={total_b/1024**3:.2f} GiB, free={free_b/1024**3:.2f} GiB")
        if "T4" not in props.name.upper():
            log(f"WARNING: GPU {i} is not a T4; FP32 SVD is still used, but timings may differ.")

    scratch_free = _free_gib(scratch)
    output_free = _free_gib(output)
    log(f"Free scratch space: {scratch_free:.1f} GiB; free output space: {output_free:.1f} GiB")
    if scratch_free < 14:
        raise RuntimeError("Need at least 14 GiB free scratch space for two streamed input shards and temporary files.")
    if output_free < 8:
        raise RuntimeError("Need at least 8 GiB free persistent output space for model, metadata, and safety margin.")

    try:
        import psutil
        ram = psutil.virtual_memory()
        log(f"System RAM: {ram.total/1024**3:.1f} GiB total, {ram.available/1024**3:.1f} GiB available")
        if ram.available < 12 * 1024**3:
            raise RuntimeError("Less than 12 GiB system RAM is available; stop background processes and retry.")
    except ImportError:
        log("psutil unavailable; skipping detailed host-RAM check.")

    return gpu_ids


def ensure_upstream(scratch: Path) -> Path:
    upstream = scratch / "SamsungLabs-LittleBit"
    if (upstream / ".git").exists():
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
        if head != UPSTREAM_COMMIT:
            run(["git", "fetch", "origin", UPSTREAM_COMMIT], cwd=str(upstream))
            run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=str(upstream))
    else:
        run(["git", "clone", UPSTREAM_REPO, str(upstream)])
        run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=str(upstream))
    # Upstream's eager quantization.utils import creates a circular import:
    # modules.littlebit -> utils.__init__ -> utils.quant_util -> modules.
    # Replace eager re-exports with a lazy, API-compatible loader. The same
    # patched file is bundled into the final model's quantization/ package.
    utils_init = upstream / "quantization" / "utils" / "__init__.py"
    utils_init.write_text(
        '"""Lazy exports for LittleBit quantization utilities.\\n'
        'Avoid importing quant_util while binary_packer is being initialised.\\n"""\\n'
        'from importlib import import_module\\n'
        '__all__ = ["get_quant_func_and_mod", "patch_inst", "apply_littlebit_patch", "binary_packer"]\\n'
        'def __getattr__(name):\\n'
        '    if name == "binary_packer":\\n'
        '        return import_module(".binary_packer", __name__).binary_packer\\n'
        '    if name in {"get_quant_func_and_mod", "patch_inst", "apply_littlebit_patch"}:\\n'
        '        return getattr(import_module(".quant_util", __name__), name)\\n'
        '    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\\n'
        'def __dir__():\\n'
        '    return sorted(set(globals()) | set(__all__))\\n',
        encoding="utf-8",
    )
    # Fail immediately, before downloading 65 GB, if the exact worker imports regress.
    probe_env = os.environ.copy()
    probe_env["PYTHONPATH"] = str(upstream) + os.pathsep + probe_env.get("PYTHONPATH", "")
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from quantization.modules.littlebit import LittleBitLinear; "
            "from quantization.functions.binary import SmoothSign; "
            "from quantization.utils import binary_packer, apply_littlebit_patch; "
            "print('LittleBit imports OK')",
        ],
        cwd=str(upstream),
        env=probe_env,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "LittleBit import smoke test failed before model download:\\n"
            + probe.stdout + probe.stderr
        )
    log(probe.stdout.strip() or "LittleBit import smoke test passed.")
    log(f"Using official SamsungLabs/LittleBit commit {UPSTREAM_COMMIT} (local circular-import compatibility patch applied)")
    return upstream


def worker_convert_shard(task: dict[str, Any]) -> dict[str, Any]:
    """Worker body: one source shard -> one LittleBit-packed safetensors shard."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    gpu_id = int(task["gpu_id"])
    torch.cuda.set_device(gpu_id)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["LOCAL_RANK"] = str(gpu_id)
    sys.path.insert(0, task["upstream_dir"])

    from quantization.modules.littlebit import LittleBitLinear
    from quantization.functions.binary import SmoothSign

    source_path = Path(task["source_path"])
    output_path = Path(task["output_path"])
    specs: dict[str, dict[str, Any]] = task["linear_specs"]
    args = task["args"]

    output_tensors: dict[str, torch.Tensor] = {}
    all_keys: list[str] = []
    converted_params = 0
    converted_bits = 0.0
    converted_layers = 0
    tensor_bytes = 0
    t0 = time.time()

    with safe_open(str(source_path), framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        key_set = set(keys)
        consumed_biases = {v["bias_key"] for v in specs.values()
                           if v.get("bias_key") and v["bias_key"] in key_set}
        for key in keys:
            # Bias is included in the converted module state alongside its factors.
            if key in consumed_biases:
                continue

            if key in specs:
                spec = specs[key]
                module_name = spec["module_name"]
                weight = sf.get_tensor(key).to(dtype=torch.float32).contiguous()
                bias_key = spec.get("bias_key")
                bias = None
                if bias_key and bias_key in key_set:
                    bias = sf.get_tensor(bias_key).to(dtype=torch.float32).contiguous()

                # Allocate only the module shell; the checkpoint tensor is assigned directly.
                module = torch.nn.Linear(
                    spec["in_features"], spec["out_features"],
                    bias=(bias is not None), device="meta", dtype=torch.float32,
                )
                module.weight = torch.nn.Parameter(weight, requires_grad=True)
                if bias is not None:
                    module.bias = torch.nn.Parameter(bias, requires_grad=False)
                module.__class__ = LittleBitLinear

                ratio = args["kv_factor"] if module_name.endswith((".k_proj", ".v_proj")) else 1.0
                module.__quant_convert__(
                    do_train=True,
                    quant_func=SmoothSign,
                    split_dim=1024,
                    eff_bit=args["eff_bit"],
                    residual=args["residual"],
                    ratio_factor=ratio,
                    min_split_dim=8,
                    use_itq=True,
                    itq_n_iter=args["itq_n_iter"],
                )

                state = module.state_dict()
                for state_key, value in state.items():
                    if value.dtype == torch.float32:
                        value = value.to(torch.bfloat16)
                    value = value.detach().cpu().contiguous().clone()
                    out_key = f"{module_name}.{state_key}"
                    output_tensors[out_key] = value
                    all_keys.append(out_key)
                    tensor_bytes += value.numel() * value.element_size()

                converted_params += int(spec["in_features"] * spec["out_features"])
                converted_bits += float(module.eff_bit_actual) * int(spec["in_features"] * spec["out_features"])
                converted_layers += 1
                del module, state, weight, bias
                torch.cuda.empty_cache()
                gc.collect()
            else:
                tensor = sf.get_tensor(key).detach().cpu().contiguous().clone()
                output_tensors[key] = tensor
                all_keys.append(key)
                tensor_bytes += tensor.numel() * tensor.element_size()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".partial")
    save_file(output_tensors, str(tmp_path), metadata={"format": "pt"})
    os.replace(tmp_path, output_path)
    duration = time.time() - t0

    if not args["keep_source_shards"]:
        try:
            source_path.unlink()
        except FileNotFoundError:
            pass

    result = {
        "output_file": output_path.name,
        "keys": all_keys,
        "linear_params": converted_params,
        "linear_effective_bits": converted_bits,
        "converted_layers": converted_layers,
        "tensor_bytes": tensor_bytes,
        "elapsed_seconds": duration,
        "run_signature": task["run_signature"],
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".partial")
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(result, f)
    os.replace(meta_tmp, meta_path)
    del output_tensors
    gc.collect()
    return result


def gpu_worker_loop(gpu_id: int, task_queue: Any, result_queue: Any) -> None:
    """Dedicated process for one GPU; prevents two simultaneous SVDs on one card."""
    try:
        import torch
        torch.cuda.set_device(gpu_id)
        while True:
            task = task_queue.get()
            if task is None:
                return
            try:
                task["gpu_id"] = gpu_id
                result = worker_convert_shard(task)
                result_queue.put((gpu_id, True, result))
            except BaseException:
                result_queue.put((gpu_id, False, traceback.format_exc()))
    except BaseException:
        result_queue.put((gpu_id, False, traceback.format_exc()))


def main() -> None:
    args = parse_args()
    if args.eff_bit <= 0 or args.eff_bit > 2:
        raise ValueError("--eff-bit must be >0 and <=2.")
    if args.kv_factor < 1:
        raise ValueError("--kv-factor must be >=1; use 1.0 for uniform rank allocation.")
    if args.workers < 1 or args.workers > 2:
        raise ValueError("--workers must be 1 or 2 on Kaggle T4 x2.")

    if not args.skip_dependency_install:
        ensure_dependencies()

    import torch
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    output_path = args.output_dir or f"/kaggle/working/littlebit2-qwen2.5-32b-{args.eff_bit:.2f}bpw"
    output_dir = Path(output_path).resolve()
    scratch_dir = Path(args.scratch_dir).resolve()
    source_dir = scratch_dir / "source"
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = resource_audit(scratch_dir, output_dir, args.workers)
    upstream_dir = ensure_upstream(scratch_dir)

    log("Downloading model config/tokenizer files (not the full ~65 GB BF16 weight set)...")
    allowed = [
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json",
        "chat_template.jinja", "chat_template.json", "LICENSE", "LICENSE.txt",
    ]
    snapshot_download(
        repo_id=args.model_id,
        repo_type="model",
        local_dir=str(output_dir),
        allow_patterns=allowed,
        token=args.hf_token,
    )
    index_path = hf_hub_download(
        repo_id=args.model_id,
        filename="model.safetensors.index.json",
        repo_type="model",
        local_dir=str(source_dir),
        token=args.hf_token,
    )
    with open(index_path, "r", encoding="utf-8") as f:
        source_index = json.load(f)
    weight_map: dict[str, str] = source_index["weight_map"]
    source_shards = sorted(set(weight_map.values()))
    log(f"Found {len(weight_map):,} checkpoint tensors across {len(source_shards)} safetensors shards.")

    config = AutoConfig.from_pretrained(str(output_dir), trust_remote_code=True)
    with init_empty_weights():
        skeleton = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    linear_specs: dict[str, dict[str, Any]] = {}
    for module_name, module in skeleton.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in module_name:
            weight_key = f"{module_name}.weight"
            bias_key = f"{module_name}.bias" if module.bias is not None else None
            linear_specs[weight_key] = {
                "module_name": module_name,
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "bias_key": bias_key,
            }

    missing = sorted(set(linear_specs) - set(weight_map))
    if missing:
        raise RuntimeError(
            f"Architecture/checkpoint mismatch: {len(missing)} expected linear weights are absent; "
            f"first missing keys: {missing[:8]}"
        )
    split_biases = [
        (wk, spec["bias_key"]) for wk, spec in linear_specs.items()
        if spec.get("bias_key") and spec["bias_key"] in weight_map
        and weight_map[spec["bias_key"]] != weight_map[wk]
    ]
    if split_biases:
        raise RuntimeError(f"Found linear biases stored in a different shard from their weights: {split_biases[:5]}")
    linear_param_count = sum(v["in_features"] * v["out_features"] for v in linear_specs.values())
    total_param_count = sum(int(p.numel()) for p in skeleton.parameters())
    if total_param_count <= 0 or linear_param_count <= 0:
        raise RuntimeError("Could not determine model parameter counts from the meta-device architecture.")
    log(f"Architecture: {len(linear_specs)} eligible linear layers; {linear_param_count/1e9:.2f}B linear weights; {total_param_count/1e9:.2f}B total parameters.")
    log(f"Compression: LittleBit-2 Joint-ITQ, SmoothSign; linear target={args.eff_bit:.3f} BPW; residual={not args.no_residual}; kv_factor={args.kv_factor:.2f}.")
    log("Embeddings, lm_head, norms, and other non-linear tensors stay at original checkpoint precision.")

    # One dedicated worker process per physical GPU. The main process feeds the
    # next shard to whichever GPU worker becomes idle; no GPU receives two SVDs.
    import multiprocessing as mp
    mp_context = mp.get_context("spawn")
    total = len(source_shards)
    completed = 0
    results: list[dict[str, Any]] = []
    start = time.time()
    task_queues = {gid: mp_context.Queue(maxsize=1) for gid in gpu_ids}
    result_queue = mp_context.Queue()
    workers = [
        mp_context.Process(target=gpu_worker_loop, args=(gid, task_queues[gid], result_queue), daemon=True)
        for gid in gpu_ids
    ]
    for proc in workers:
        proc.start()
    available_gpu_ids = list(gpu_ids)
    pending = 0
    download_pool = None
    download_future = None
    download_future_index = None
    run_signature = {
        "model_id": args.model_id,
        "eff_bit": float(args.eff_bit),
        "kv_factor": float(args.kv_factor),
        "itq_n_iter": int(args.itq_n_iter),
        "residual": not args.no_residual,
        "upstream_commit": UPSTREAM_COMMIT,
    }

    def collect_one() -> None:
        nonlocal pending, completed
        try:
            finished_gpu, ok, payload = result_queue.get(timeout=4 * 3600)
        except Exception as exc:
            dead = [p for p in workers if not p.is_alive()]
            raise RuntimeError(f"Timed out waiting for GPU worker; dead workers={[(p.pid, p.exitcode) for p in dead]}") from exc
        pending -= 1
        available_gpu_ids.append(finished_gpu)
        if not ok:
            raise RuntimeError(f"GPU {finished_gpu} worker failed:\n{payload}")
        results.append(payload)
        completed += 1
        log(f"Converted shard {completed}/{total}: {payload['output_file']} | "
            f"{payload['converted_layers']} linear layers | {payload['elapsed_seconds']/60:.1f} min")

    def output_is_complete(shard_number: int) -> bool:
        name = f"model-{shard_number:05d}-of-{total:05d}.safetensors"
        path = output_dir / name
        return path.exists() and path.with_suffix(path.suffix + ".meta.json").exists()

    def download_one(shard_name: str) -> str:
        log(f"Prefetching source shard: {shard_name}")
        return hf_hub_download(
            repo_id=args.model_id,
            filename=shard_name,
            repo_type="model",
            local_dir=str(source_dir),
            token=args.hf_token,
        )

    def next_missing_index(after_index: int) -> int | None:
        for candidate in range(after_index + 1, total + 1):
            if not output_is_complete(candidate):
                return candidate
        return None

    try:
        from concurrent.futures import ThreadPoolExecutor
        download_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-prefetch")

        for shard_index, shard_name in enumerate(source_shards, start=1):
            out_name = f"model-{shard_index:05d}-of-{total:05d}.safetensors"
            out_path = output_dir / out_name
            meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
            # Safe restart after a Kaggle 12-hour session cutoff: completed shards
            # are persistent under /kaggle/working and are not recomputed.
            if out_path.exists() and meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    resumed = json.load(f)
                if resumed.get("run_signature") != run_signature:
                    raise RuntimeError(
                        f"Existing shard {out_name} was produced with different settings. "
                        "Choose a fresh --output-dir or restore the matching BPW/KV/residual settings."
                    )
                results.append(resumed)
                completed += 1
                log(f"Resuming: shard {shard_index}/{total} already completed ({out_name}).")
                continue
            if out_path.exists() and not meta_path.exists():
                out_path.unlink()  # orphaned file from interruption before sidecar write

            if not available_gpu_ids:
                # The next source shard is already downloading on a background thread.
                collect_one()
            gpu_id = available_gpu_ids.pop(0)

            if download_future is None or download_future_index != shard_index:
                download_future = download_pool.submit(download_one, shard_name)
                download_future_index = shard_index
            local_source = download_future.result()
            download_future = None
            download_future_index = None

            # Prefetch exactly one upcoming missing shard while this shard is
            # being factorized. This keeps scratch usage bounded to ~2 source shards.
            lookahead = next_missing_index(shard_index)
            if lookahead is not None:
                download_future = download_pool.submit(download_one, source_shards[lookahead - 1])
                download_future_index = lookahead

            task = {
                "source_path": str(local_source),
                "output_path": str(out_path),
                "linear_specs": linear_specs,
                "upstream_dir": str(upstream_dir),
                "run_signature": run_signature,
                "args": {
                    "eff_bit": float(args.eff_bit),
                    "kv_factor": float(args.kv_factor),
                    "itq_n_iter": int(args.itq_n_iter),
                    "residual": not args.no_residual,
                    "keep_source_shards": bool(args.keep_source_shards),
                },
            }
            task_queues[gpu_id].put(task)
            pending += 1

        while pending:
            collect_one()
    finally:
        if download_pool is not None:
            download_pool.shutdown(wait=True, cancel_futures=True)
        for q in task_queues.values():
            try:
                q.put(None)
            except Exception:
                pass
        for proc in workers:
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)

    if completed != total:
        raise RuntimeError(f"Incomplete conversion: {completed}/{total} shards completed.")

    # Build a standard sharded index for the official LittleBit loader.
    output_weight_map: dict[str, str] = {}
    total_tensor_bytes = 0
    total_linear_params = 0
    total_linear_bits = 0.0
    total_layers = 0
    for result in results:
        for key in result["keys"]:
            if key in output_weight_map:
                raise RuntimeError(f"Duplicate output tensor key: {key}")
            output_weight_map[key] = result["output_file"]
        total_tensor_bytes += int(result["tensor_bytes"])
        total_linear_params += int(result["linear_params"])
        total_linear_bits += float(result["linear_effective_bits"])
        total_layers += int(result["converted_layers"])

    if not output_weight_map:
        raise RuntimeError("No output tensors were produced.")

    index = {
        "metadata": {"total_size": total_tensor_bytes},
        "weight_map": output_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    quant_config = {
        "quant_func": "SmoothSign",
        "quant_mod": "LittleBitLinear",
        "eff_bit": float(args.eff_bit),
        "split_dim": 1024,
        "residual": not args.no_residual,
        "kv_factor": float(args.kv_factor),
        "min_split_dim": 8,
        "use_itq": True,
        "itq_n_iter": int(args.itq_n_iter),
        "qat_trained": False,
        "method": "LittleBit-2 SVD + Joint-ITQ initialization only",
    }
    with open(output_dir / "littlebit_config.json", "w", encoding="utf-8") as f:
        json.dump(quant_config, f, indent=2)

    config_path = output_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config_json = json.load(f)
    config_json.update({k: v for k, v in quant_config.items() if k not in ("method", "qat_trained")})
    config_json["auto_map"] = {"AutoModelForCausalLM": "quantization.hub.LittleBitModel"}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)

    # Ship the small upstream loader package with the checkpoint so the artifact
    # remains self-describing and can be loaded without a separate source checkout.
    shutil.copytree(upstream_dir / "quantization", output_dir / "quantization", dirs_exist_ok=True)
    shutil.copy2(upstream_dir / "LICENSE", output_dir / "LITTLEBIT_LICENSE")

    file_size = sum(p.stat().st_size for p in output_dir.glob("model-*.safetensors"))
    elapsed = time.time() - start
    linear_bpw_actual = total_linear_bits / max(1, total_linear_params)
    whole_model_bpw_est = (
        total_linear_bits + (total_param_count - total_linear_params) * 16.0
    ) / total_param_count
    file_bpw = file_size * 8.0 / total_param_count

    report = {
        "model_id": args.model_id,
        "official_littlebit_repo": UPSTREAM_REPO,
        "official_littlebit_commit": UPSTREAM_COMMIT,
        "littlebit2_joint_itq": True,
        "quant_func": "SmoothSign",
        "target_linear_bpw": float(args.eff_bit),
        "measured_linear_bpw_from_layer_configs": linear_bpw_actual,
        "estimated_whole_model_bpw_including_16bit_non_linears": whole_model_bpw_est,
        "on_disk_safetensors_bpw_including_headers": file_bpw,
        "total_parameters": total_param_count,
        "quantized_linear_parameters": total_linear_params,
        "converted_linear_layers": total_layers,
        "safetensors_tensor_bytes": total_tensor_bytes,
        "safetensors_file_bytes": file_size,
        "qat_trained": False,
        "quality_loss_guarantee": False,
        "elapsed_seconds": elapsed,
        "notes": [
            "SVD + Joint-ITQ initialization only; no teacher/student distillation or QAT was run.",
            f"The configured {args.eff_bit:.3f} BPW target applies to eligible linear weights, not all model parameters.",
            "Embeddings and lm_head remain at source precision and materially increase whole-model BPW.",
            "LittleBit binary factors are packed on disk; the official loader expands them for inference.",
        ],
    }
    with open(output_dir / "COMPRESSION_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    readme = f"""# LittleBit-2 Qwen2.5-32B-Instruct (Sarvagya build)

This checkpoint was generated by the Sarvagya Kaggle streaming converter using the official SamsungLabs/LittleBit implementation at commit `{UPSTREAM_COMMIT}`.

## Configuration

- Method: SVD + LittleBit-2 Joint-ITQ + SmoothSign binary factors
- Linear target: `{args.eff_bit:.3f}` BPW
- Residual path: `{not args.no_residual}`
- KV rank multiplier: `{args.kv_factor:.2f}`
- Joint-ITQ iterations: `{args.itq_n_iter}`
- Teacher/student QAT: **not performed**

## Important quality note

This is an initialization-only compressed checkpoint. It is not lossless, and no no-quality-loss claim is made. For best task fidelity, run the official teacher/student QAT recipe on substantially larger hardware and compare perplexity and task benchmarks against the BF16 source model.

The target BPW applies to eligible linear weights only. Embeddings, `lm_head`, normalization parameters, and other non-linear tensors are retained at source precision. See `COMPRESSION_REPORT.json` for measured effective bit rates and sizes.

## Loading / memory note

The output is compact on disk. The current official loader expands packed binary factors to floating-point tensors for inference; this can require substantially more RAM/VRAM than the checkpoint file size. A pair of T4 GPUs does not present one pooled 30 GiB device, and the official loader may not fit this 32B model on a single T4. Use the bundled `quantization/` loader package on a host with sufficient memory, or implement a packed-factor inference kernel. Ordinary `AutoModelForCausalLM.from_pretrained()` is not the intended loader for this checkpoint.

## Licensing

The LittleBit implementation is CC BY-NC 4.0. Check the base Qwen model license and all third-party terms before redistribution or commercial use.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    log("COMPLETE")
    log(f"Output: {output_dir}")
    log(f"Compressed safetensors: {file_size/1024**3:.2f} GiB")
    log(f"Linear BPW actual (weighted): {linear_bpw_actual:.4f}")
    log(f"Estimated whole-model BPW: {whole_model_bpw_est:.4f}")
    log(f"Whole checkpoint disk BPW (headers included): {file_bpw:.4f}")
    log(f"Elapsed: {elapsed/3600:.2f} hours")
    log("WARNING: no QAT/distillation was run; quality loss is possible and must be benchmarked.")

    if not args.keep_source_shards:
        try:
            shutil.rmtree(source_dir)
        except OSError as exc:
            log(f"Scratch cleanup warning: {exc}")


if __name__ == "__main__":
    main()
