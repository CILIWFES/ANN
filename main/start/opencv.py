import cv2
import math
import numpy as np
from main.dataprocessing.image import *
from main.analysis.performanceMeasure import *

picture = IMP.readPicture('E:\\', "sda.jpg")
channels = IMP.conversionChannels(picture)

IMP.showPicture_RBG(picture)

# IMP.showGrayscale(channels, np.array(['B', 'G', 'R']))

# cv2.imshow('picture', picture)

# cv2.waitKey(0)
bb = IMP.cutOut(channels, (120, 200), 100, 200)
IMP.showPicture_BRG(IMP.reversalRGB(picture), wait=True)
