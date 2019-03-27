from main.frame.mxnet import *
import numpy as np
from main.dataprocessing.image import *

width = 100
hight = 40
model = MX_ORM.Module_Read('verification/VG', (1, 3, hight, width), labelShape=(1, 4), epoch=15)
path = 'E:/CodeSpace/Python/ANN/files/training-package/verification/make/'
name = 'test.png'


def reload(path, name, toWidth=100, toHigh=40):
    image = IMP.resize(IMP.readChannels(path, name), toWidth=toWidth, toHigh=toHigh)
    IMP.showPicture_RBG(IMP.conversionChannels(image, False))
    print(np.argmax(MX_Prediction.prediction(np.array([image]), model), axis=1))


reload(path, name, width, hight)


def make(toWidth=100, toHigh=40):
    image = IMP.resize(IMP.readChannels(path, name), toWidth=toWidth, toHigh=toHigh)
    IMP.showPicture_RBG(IMP.conversionChannels(image, False))
    print(np.argmax(MX_Prediction.prediction(np.array([image]), model), axis=1))
