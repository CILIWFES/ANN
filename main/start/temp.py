from main.dataprocessing import *
import numpy as np

# (trainImages, trainLabels), (testImages, testLabels) = VG.loadSet()
VG.makeSet((40, 20), 9000, 900, codeSize=2)
# print((np.array([12, 55, 88]) == np.array([55, 88, 12])).sum())
