# Data

Intermediate JSON results from the benchmarking notebook.

- `systems_partial.json`: aggregated metrics for FP16 baseline and speculative k=3
- `profile_fp16.json`: PyTorch profiler trace for 10 FP16 decode steps

The full 400-run dataset (`rich_data_a100.json`) is regenerable from `notebooks/msml605_systems.ipynb` and is large enough to be distributed via release rather than committed.
