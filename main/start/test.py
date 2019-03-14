import pickle
import numpy as mp
from main.dataprocessing.image import *
from main.dataprocessing.training_set import *


def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


# data = load_file(r'D:\CodeSpace\Python\ANN\files\training-package\cifar-10\batches.meta')
# 32*32

# print(data.keys())
Cifar.toConversion()

