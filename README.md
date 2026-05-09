# Systems-Level Benchmarking of LLM Inference

Final project for **MSML 605: Computing Systems for Machine Learning**, University of Maryland, College Park (Spring 2026).

**Author:** Pratham Ramachandra Bharati  
**Report:** [docs/final_report.pdf](docs/final_report.pdf)

## Summary

We benchmark autoregressive decoding for Qwen 2.5 7B Instruct against speculative decoding with a Qwen 2.5 0.5B draft, sweeping speculation length from k=1 to k=10 (400 measured runs). The headline finding: **speculative decoding does not produce a wall-clock speedup at any value of k** on a single A100 80GB at batch size 1. The FP16 baseline reaches 26.4 tok/s; the best speculative configuration (k=3) reaches only 14.2 tok/s, a 0.54x slowdown. We diagnose this with a roofline analysis and show the textbook formula predicts the slowdown to within 1%.

## Key results

| Configuration | Throughput | Memory | Notes |
|---|---|---|---|
| FP16 baseline | 26.4 tok/s | 17.0 GB | Reference point |
| Speculative k=3 (best) | 14.2 tok/s | 17.4 GB | 0.54x of FP16 |
| Speculative k=1 | 12.8 tok/s | 17.4 GB | Acceptance 0.767 |
| Speculative k=10 | 10.85 tok/s | 17.4 GB | Acceptance 0.356 |

## Repository layout

```
.
├── README.md                 This file
├── Dockerfile                Reproducible environment (CUDA 12.1 + PyTorch)
├── docker-compose.yml        One-command launch
├── requirements.txt          Python dependencies
├── docs/
│   └── final_report.pdf      4-page IEEE-format report
├── notebooks/
│   └── msml605_systems.ipynb Full benchmarking pipeline
├── figures/                  All plots referenced in the report
└── data/                     Intermediate measurement results
```

---

## How to run

There are **three options** for running this project. Pick whichever matches your setup.

### Option 1: Docker (recommended, reproducible)

**Prerequisites:**
- Docker Desktop installed ([download](https://www.docker.com/products/docker-desktop/))
- NVIDIA Container Toolkit ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- An NVIDIA GPU with CUDA 12.1+ support (A100 used in this work; H100 / A6000 / RTX 4090 should also work)

**To pull and run the pre-built image from Docker Hub:**

```bash
docker pull prathambharati/msml605-bharati:latest
docker run --gpus all -p 8888:8888 prathambharati/msml605-bharati:latest
```

Then open `http://localhost:8888` in your browser. The notebook is at `notebooks/msml605_systems.ipynb`.

**To build the image yourself from source:**

```bash
git clone https://github.com/prathambharati/msml605-systems-benchmarking.git
cd msml605-systems-benchmarking
docker build -t msml605-bharati:latest .
docker run --gpus all -p 8888:8888 msml605-bharati:latest
```

**Or use docker-compose:**

```bash
docker-compose up
```

### Option 2: Google Colab Pro (matches our setup)

The benchmarking notebook was developed and validated on Google Colab Pro with an A100 instance. To reproduce:

1. Upload `notebooks/msml605_systems.ipynb` to Colab
2. Set the runtime to A100 GPU (Runtime → Change runtime type → A100)
3. Run all cells sequentially. The FP16 baseline takes about 5 minutes; the speculative k-sweep takes about 90 minutes for 400 runs

### Option 3: Local installation

**Prerequisites:**
- Python 3.10+
- An NVIDIA GPU with CUDA 12.1+ (A100 80GB recommended; smaller GPUs will fail to load Qwen 2.5 7B in FP16)
- ~30 GB free disk space (for model downloads)

```bash
git clone https://github.com/prathambharati/msml605-systems-benchmarking.git
cd msml605-systems-benchmarking

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/msml605_systems.ipynb
```

---

## Hardware caveats

- **A100 80GB** is what we used. The 7B model in FP16 needs ~14 GB for weights plus ~3 GB for KV cache, so it fits comfortably with overhead for the draft model.
- **Smaller GPUs** (T4, A10, RTX 4090 24GB) will run out of memory loading both the 7B target and 0.5B draft together. To run on a smaller GPU, you can:
  - Run only the FP16 baseline cells
  - Use a smaller target model (Qwen 2.5 1.5B)
  - Apply quantization (which we attempted but were blocked on by a bitsandbytes/Triton issue, see Section VI of the report)

---

## What we attempted but could not complete

The original project plan included INT8 and INT4 (NF4) quantization through bitsandbytes. The implementation is in the notebook but execution was blocked by a binary incompatibility between bitsandbytes 0.43.1 and the Triton compiler version that ships with Colab's PyTorch image. The specific failure is `ModuleNotFoundError: No module named 'triton.ops'` at model-load time. Three workarounds were attempted (version pinning, manual Triton install, device_map variations); none resolved the issue cleanly within the project timeline.

The report (`docs/final_report.pdf`) presents the FP16 + speculative comparison plus the roofline analysis, and discusses what the roofline predicts INT8/INT4 should achieve based on the bytes-per-parameter reduction. Quantization measurement is left as future work, ideally on a self-hosted environment with full control over the bitsandbytes/Triton stack.

---

## Citation

```bibtex
@misc{bharati2026systems,
  author       = {Pratham Ramachandra Bharati},
  title        = {Systems-Level Benchmarking of LLM Inference: Why Speculative Decoding Does Not Help on a Single A100 GPU},
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
