"""
Testing machine learning methods for converting from 10m to 100m wind speed.

Based on: https://www.sciencedirect.com/science/article/pii/S1364032122007791#da0010

With help from: https://www.blog.trainindata.com/lasso-feature-selection-with-python/
"""

# imports
import os
import time
import sys
import glob

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim