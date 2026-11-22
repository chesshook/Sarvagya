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

- LittleBit-2 Joint-ITQ enabled; 50 ITQ iterations
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

## Critical limitations

- **This is not lossless.** No low-bit weight approximation can guarantee identical outputs to the BF16 source checkpoint.
- **No QAT or teacher/student distillation is run.** The official SamsungLabs training recipe includes QAT; a full 32B teacher/student training run is not practical on Kaggle T4 x2 within the session and RAM limits. This script performs the official SVD + Joint-ITQ initialization and binary packing only.
- The two T4 GPUs do not form one pooled 30 GiB device. The conversion is parallelized with one pinned worker per card, but inference placement is a separate problem.
- The upstream loader expands packed binary factors to floating-point tensors for inference. The compact on-disk size therefore does not mean the model will fit in 15 GiB VRAM. Use a machine with sufficient host/GPU memory or a custom packed-factor inference implementation.
- Completed shard files and matching metadata sidecars are retained so the script can resume or skip already-finished shards when rerun with the same settings. It refuses to mix shards produced with different quantization settings.

## Expected runtime

Plan for approximately **4–10 hours** for a first conversion, subject to Hugging Face download throughput and Kaggle T4 performance. This is an engineering estimate, not a measured benchmark; it can exceed one 12-hour Kaggle session. The script logs per-shard time and supports resuming completed shards when the output directory is preserved. The full 32B QAT/distillation stage is not included and would require substantially more capable hardware.

## Upstream and licensing

- LittleBit-2 implementation: [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit)
- Pinned upstream commit: `42d658b0c79f76450b34b6a3547462c7cdc3e1a0`
- Base model: [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)

The upstream LittleBit implementation is licensed **CC BY-NC 4.0**. The base model and all dependencies have their own terms; review these before redistribution or commercial use.
