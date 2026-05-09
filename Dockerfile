# Dockerfile for MSML 605 final project
# Pratham Ramachandra Bharati - University of Maryland, College Park
#
# This image provides a reproducible environment for running the LLM inference
# benchmarking notebook. Based on NVIDIA's official PyTorch image with CUDA 12.1.

FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL maintainer="pratham.bharati03@gmail.com"
LABEL description="MSML 605: Systems-level benchmarking of LLM inference"

# Set working directory
WORKDIR /workspace

# Install Python dependencies
# Using --break-system-packages because the NVIDIA image uses system Python
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# Copy project files
COPY notebooks/ /workspace/notebooks/
COPY figures/ /workspace/figures/
COPY data/ /workspace/data/
COPY docs/ /workspace/docs/
COPY README.md /workspace/

# Expose Jupyter port
EXPOSE 8888

# Default command: launch Jupyter notebook server
# Access at http://localhost:8888 with no token (development setup)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", \
     "--NotebookApp.token=''", "--NotebookApp.password=''", \
     "--notebook-dir=/workspace"]
