import glob
from random import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from random import shuffle, randint

targets = []
features = []

files = glob.glob("datasets/*.jpg")
shuffle(files)

for file in tqdm(files[:500]):
    features.append(np.array(Image.open(file).resize((75, 75))))
    target = [1, 0] if "cat" in file else [0, 1]
    targets.append(target)

features = np.array(features)
targets = np.array(targets)

print(f"Shape features: {features.shape}")
print(f"Shape targets: {targets.shape}")
