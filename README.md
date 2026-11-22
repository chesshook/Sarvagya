# Sarvagya — Kaggle LittleBit-2 Qwen2.5-32B

This repository contains a streaming converter for **Qwen/Qwen2.5-32B-Instruct** using the official **SamsungLabs/LittleBit** implementation and its LittleBit-2 Joint-ITQ initialization.

## Run on Kaggle

1. Create a Kaggle Notebook and set **Accelerator: GPU T4 x2** and **Internet: On**.
2. Run this single notebook cell:

```bash
!if [ -d /kaggle/working/Sarvagya/.git ]; then git -C /kaggle/working/Sarvagya pull --ff-only; else git clone -q https://github.com/chesshook/Sarvagya.git /kaggle/working/Sarvagya; fi && python /kaggle/working/Sarvagya/kaggle/littlebit2_qwen25_32b.py
```

The script installs the small set of Python dependencies it needs while retaining Kaggle's preinstalled CUDA-enabled PyTorch. It downloads the source checkpoint shard-by-shard, pins one worker process to each T4, writes the compressed checkpoint into `/kaggle/working/`, and removes each source shard from temporary storage after conversion.

Default output directory:

```
/kaggle/working/littlebit2-qwen2.5-32b-0.45bpw/
```

Kaggle must have enough free scratch space for two source shards and enough persistent working space for the finished checkpoint. The script checks both before downloading the model.

## Compression settings

The default profile is intentionally less aggressive than the requested 0.30 BPW:

- LittleBit-2 Joint-ITQ enabled; 10 ITQ iterations by default (use `--itq-n-iter 50` for slower, more thorough alignment)
- SmoothSign binary factors
- Residual factorization enabled
- 0.45 target BPW for eligible linear weight matrices
- 1.15 rank multiplier for key/value projections
- Embeddings, `lm_head`, norms, and other non-linear tensors kept at source precision

To force the user's requested 0.30 BPW instead, append:

```bash
--eff-bit 0.30
```

The BPW target applies to eligible `nn.Linear` weights, not the entire model. The script reports measured linear BPW, estimated whole-model BPW, and actual checkpoint file BPW in `COMPRESSION_REPORT.json`.

## Recent Kaggle compatibility fix

The converter applies a small, API-compatible lazy-import patch to the pinned upstream `quantization.utils` package before loading workers. The upstream eager `utils/__init__.py` re-export caused a circular import when importing `LittleBitLinear`. A smoke test now runs before any model weights are downloaded, so this class of import failure is caught immediately.

Dependency installation now uses `pip --no-deps` and only installs missing runtime packages. This avoids changing Kaggle's preinstalled PyTorch/CUDA and Hugging Face Hub stack merely to satisfy unrelated Gradio/Diffusers resolver constraints. Pip resolver warnings from the old run were not the cause of the crash.

**After a failed run:** pull the latest Sarvagya commit and rerun the same command in the same Kaggle session. The first few source shards already fetched under `/kaggle/temp/sarvagya_littlebit2/source` should be reused from the Hugging Face local cache; no output shards had been committed before the circular-import failure.

## Critical limitations

- **This is not lossless.** No low-bit weight approximation can guarantee identical outputs to the BF16 source checkpoint.
- **No QAT or teacher/student distillation is run.** The official SamsungLabs training recipe includes QAT; a full 32B teacher/student training run is not practical on Kaggle T4 x2 within the session and RAM limits. This script performs the official SVD + Joint-ITQ initialization and binary packing only.
- The two T4 GPUs do not form one pooled 30 GiB device. The conversion is parallelized with one pinned worker per card, but inference placement is a separate problem.
- The upstream loader expands packed binary factors to floating-point tensors for inference. The compact on-disk size therefore does not mean the model will fit in 15 GiB VRAM. Use a machine with sufficient host/GPU memory or a custom packed-factor inference implementation.
- Completed shard files and matching metadata sidecars are retained so the script can resume or skip already-finished shards when rerun with the same settings. It refuses to mix shards produced with different quantization settings.

## Fast runtime profile

The default script is tuned to make a **1–2 hour run plausible**, not guaranteed:

- 10 Joint-ITQ iterations instead of 50
- One-file asynchronous download prefetch while the two GPU workers quantize
- Two persistent, GPU-pinned workers, one per T4
- 0.45 BPW with residual factorization retained

The base weights are **65.5 GB**. To download 65.5 GB in two hours requires at least about **9.1 MB/s sustained** before conversion overhead; in one hour it requires about **18.2 MB/s**. For a practical 1–2 hour end-to-end target, aim for roughly 15–25 MB/s or better from Hugging Face. Actual Kaggle network speed, T4 utilization, and storage throughput vary, so this is a target, not a benchmarked guarantee.

If time matters more than quality, use a more aggressive run:

```bash
python /kaggle/working/Sarvagya/kaggle/littlebit2_qwen25_32b.py --eff-bit 0.30 --itq-n-iter 8
```

That lowers the factor rank and should reduce decomposition time, but it increases approximation error. It is not the profile I recommend for preserving intelligence.

The script logs per-shard times and supports resuming completed shards when the output directory is preserved. The full 32B QAT/distillation stage is not included and would require substantially more capable hardware.

## Upstream and licensing

- LittleBit-2 implementation: [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit)
- Pinned upstream commit: `42d658b0c79f76450b34b6a3547462c7cdc3e1a0`
- Base model: [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)

The upstream LittleBit implementation is licensed **CC BY-NC 4.0**. The base model and all dependencies have their own terms; review these before redistribution or commercial use.
