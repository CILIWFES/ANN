from main.dataprocessing import *

# (trainImages, trainLabels), (testImages, testLabels) = VG.loadSet()
VG.makeSet((40, 20), 9000, 900, codeSize=2)
