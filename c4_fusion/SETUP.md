# Local GPU Environment Setup (c4_fusion)

This project supports an optional CUDA-enabled Python environment for GPU workflows.

## Dependencies
- Install with:
  python -m pip install -r c4_fusion/requirements-gpu.txt

## Activating the Environment
The local environment is created at `.venv` with Python 3.11.9.

To activate:

- On PowerShell:
  .\.venv\Scripts\Activate.ps1

- On Command Prompt:
  .\.venv\Scripts\activate.bat

- On bash (Linux/macOS):
  source .venv/bin/activate

## Notes
- `.venv/` is already ignored in `.gitignore`.
- `requirements-gpu.txt` records CUDA-enabled PyTorch build and geospatial/Jupyter dependencies.
- Use this environment only if you have an NVIDIA GPU available.
