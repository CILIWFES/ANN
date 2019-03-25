from main.frame.mxnet import *
import numpy as np
from main.dataprocessing.image import *

model = MX_ORM.Module_Read('test/simple', (1, 3, 20, 40), labelShape=(1, 2), epoch=96)
path = 'D:/CodeSpace/Python/ANN/files/training-package/verification/train/'
name = '47_SitkaZ.ttc.png'


def reload(path, name, toWidth=40, toHigh=20):
    image = IMP.resize(IMP.readChannels(path, name), toWidth=toWidth, toHigh=toHigh)
    IMP.showPicture_RBG(IMP.conversionChannels(image, False))
    print(np.argmax(MX_Prediction.prediction(np.array([image]), model), axis=1))


reload(path, name)
