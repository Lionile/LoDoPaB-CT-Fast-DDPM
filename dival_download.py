import numpy as np
import torch
from dival import get_standard_dataset
from dival.datasets import LoDoPaBDataset

dataset = get_standard_dataset('lodopab')

print(type(dataset))
