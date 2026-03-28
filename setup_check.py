import torch
import cv2
import nibabel
import numpy as np

print(f"PyTorch versão: {torch.__version__}")
print(f"OpenCV versão: {cv2.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}") # Se for True, você pode usar a GPU!