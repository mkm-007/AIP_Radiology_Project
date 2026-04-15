# scripts/test_environment.py

import sys
import os
import torch
import pandas as pd
import numpy as np
import sklearn
import cv2
import matplotlib

print("=== Environment Test ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print("Environment setup looks OK.")