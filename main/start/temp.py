from main.dataprocessing import *
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter




# (trainImages, trainLabels), (testImages, testLabels) = VG.loadSet()
VG.makeSet((100, 40), 2000, 200, codeSize=4)
# print((np.array([12, 55, 88]) == np.array([55, 88, 12])).sum())
# img = IMP.readPicture('D:/CodeSpace/Python/ANN/files/training-package/verification/train/', 'PM08_seguili.ttf.png')
# img = get_img(img)
# IMP.showPicture_RBG(img)
