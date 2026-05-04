# Why Speculative Decoding Does Not Help

Systems-level benchmarking and roofline analysis of LLM inference acceleration techniques on a single NVIDIA A100 80GB GPU. Final project for **MSML 605: Computing Systems for Machine Learning**, University of Maryland, College Park (Spring 2026).

## TL;DR

We benchmark autoregressive decoding for Qwen 2.5 7B Instruct against speculative decoding with a Qwen 2.5 0.5B draft, sweeping speculation length from k=1 to k=10 (400 measured runs). The headline finding: **speculative decoding does not produce a wall-clock speedup at any value of k**. The FP16 baseline reaches 26.4 tok/s; the best speculative configuration (k=3) reaches only 14.2 tok/s, a 0.54x slowdown. We diagnose this with a roofline analysis and show the textbook formula predicts the slowdown to within 1%.

## Key results

| Configuration | Throughput | Memory | Notes |
|---|---|---|---|
| FP16 baseline | 26.4 tok/s | 17.0 GB | Reference |
| Speculative k=3 (best) | 14.2 tok/s | 17.4 GB | 0.54x of FP16 |
| Speculative k=1 | 12.8 tok/s | 17.4 GB | Acceptance 0.767 |
| Speculative k=10 | 10.85 tok/s | 17.4 GB | Acceptance 0.356 |

## Why speculative decoding fails on this hardware

The textbook speedup formula predicts speedup as a function of acceptance probability and the per-token time ratio between draft and target. With our measured alpha = 0.629 and c = 0.82 (1.22x ratio), the predicted speedup at k=3 is 0.55x. The empirical speedup is 0.54x. The formula is right; the assumption that the draft is "much faster" simply does not hold at batch=1 on memory-bound hardware.

The deeper reason: A100 has a ridge point at 156 FLOPs/byte. Single-token decode operates at ~2 FLOPs/byte. Both target and draft are deeply memory-bound. Per-token weight-read time (7 ms for the target, 0.5 ms for the draft) is dwarfed by fixed per-token overheads (kernel launches, attention, KV cache). The 14x parameter gap collapses to a 1.22x time gap in practice.

## Repository layout

```
.
├── README.md                       This file
├── docs/
│   └── final_report.pdf            10-page IEEE-style report
├── notebooks/
│   └── msml605_systems.ipynb       Full benchmarking pipeline
├── figures/
│   ├── fig_roofline.png            A100 roofline with operating points
│   ├── fig_k_sweep.png             Throughput across k=1-10 (400 runs)
│   ├── fig_acceptance.png          Per-round acceptance vs k
│   ├── fig_latency.png             FP16 latency distribution (200 steps)
│   └── fig_kernels.png             PyTorch profiler kernel breakdown
└── data/
    └── (intermediate JSON results from benchmark runs)
```

## Reproducing

The benchmarking notebook runs end-to-end on Google Colab Pro with an A100 instance.

1. Open `notebooks/msml605_systems.ipynb` in Colab
2. Set the runtime to A100 GPU
3. Run cells sequentially; the FP16 baseline takes about 5 minutes, the speculative k-sweep takes about 90 minutes for 400 runs

**Hardware needed.** NVIDIA A100 80GB. Smaller GPUs (T4, A10) will run out of memory loading Qwen 2.5 7B in FP16 alongside the 0.5B draft.

## What we attempted but could not complete

The original project plan included INT8 and INT4 (NF4) quantization through bitsandbytes. The implementation is in the notebook but execution was blocked by a binary incompatibility between bitsandbytes 0.43.1 and the Triton compiler version that ships with Colab's PyTorch image. The specific failure is `ModuleNotFoundError: No module named 'triton.ops'` at model-load time. Three workarounds were attempted (version pinning, manual Triton install, device_map variations); none resolved the issue cleanly within the project timeline.

The final report (`docs/final_report.pdf`) presents the FP16 + speculative comparison plus the roofline analysis, and discusses what the roofline predicts INT8/INT4 should achieve based on the bytes-per-parameter reduction. Quantization measurement is left as future work, ideally on a self-hosted environment with full control over the bitsandbytes/Triton stack.

## Hardware regime takeaway

The roofline framework lets us predict where each technique helps:

**Memory-bound regime (this work):** quantization is the right answer. Reducing bytes per parameter directly reduces HBM traffic, which is the actual bottleneck.

**Compute-bound regime (not tested here):** speculative decoding makes sense. This is the regime of larger batch sizes, smaller draft models, or specialized speculative pipelines like Medusa and EAGLE.

The practical recommendation: run a roofline analysis before committing to any acceleration technique. Match the technique to the regime, not to the popularity of the technique in the literature.

## Citation

```bibtex
@misc{bharati2026why,
  author       = {Pratham Ramachandra Bharati},
  title        = {Why Speculative Decoding Does Not Help: A Systems-Level Roofline Analysis of LLM Inference on a Single A100 GPU},
  year         = {2026},
  howpublished = {Final project, MSML 605, University of Maryland, College Park},
  url          = {https://github.com/prathambharati/msml605-systems-benchmarking}
}
```

## Contact

Pratham Ramachandra Bharati  
M.S. Applied Machine Learning, University of Maryland, College Park  
pratham.bharati03@gmail.com

## License

MIT
