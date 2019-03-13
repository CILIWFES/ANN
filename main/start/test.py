import pickle
import numpy as mp
from main.dataprocessing.image import *


def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


data = load_file('C:\\Users\\ZC\\Downloads\\cifar-10-batches-py\\batches.meta')
# 32*32

print(data.keys())
