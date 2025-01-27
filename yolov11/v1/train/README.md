
# YOLO Finetuning Environment Setup

This README provides step-by-step instructions to set up the environment for fine-tuning YOLO (You Only Look Once) models.

## Prerequisites

Ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your system.

## Environment Setup

1. **Create a Conda Environment**

   ```bash
   conda create --name yolo-finetune python=3.11
   ```
2. **Activate the Environment**

   ```bash
   conda activate yolo-finetune
   ```
3. **Install Required Packages**

   - Install Matplotlib:
     ```bash
     conda install matplotlib
     ```
   - Install PyTorch and Related Libraries:
     ```bash
     conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
     ```
4. **Verify Installation**
   After installation, you can verify that PyTorch and CUDA are set up correctly:

   ```python
   import torch
   print("PyTorch version:", torch.__version__)
   print("CUDA available:", torch.cuda.is_available())
   print("CUDA version:", torch.version.cuda)
   print("Available GPUs:", torch.cuda.device_count())
   if torch.cuda.is_available():
       for i in range(torch.cuda.device_count()):
           print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
   ```
5. **Check GPU Details with NVIDIA-SMI**
   To get detailed information about your GPU, use the NVIDIA System Management Interface:

   ```bash
   nvidia-smi
   ```

   This will display details such as GPU utilization, memory usage, and driver versions.

## Notes

- Make sure your system has a compatible NVIDIA GPU and the necessary drivers to support CUDA 12.4.
- For further details on PyTorch installation, visit the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Troubleshooting

- If you encounter issues during installation, consider updating Conda:
  ```bash
  conda update conda
  ```
- Verify the installed packages:
  ```bash
  conda list
  ```
