from main.frame.mxnet import *
import numpy as np
from main.dataprocessing.image import *

model = MX_ORM.Module_Read('test2', (1, 3, 28, 28), epoch=26)
path = 'D:/CodeSpace/Python/ANN/files/temp/'
name = 'fj.jpg'


def reload(path, name):
    image = IMP.resize(IMP.readChannels(path, name), toWidth=28, toHigh=28)
    IMP.showPicture_RBG(IMP.conversionChannels(image, False))
    print(np.argmax(MX_Prediction.prediction(np.array([image]), model)))


reload(path, name)
