from main.frame.mxnet import *
import numpy as np
import random
from main.dataprocessing import *

width = 100
hight = 40
model = MX_ORM.Module_Read('verification/VG', (1, 1, hight, width), labelShape=(1, 4), epoch=102)
path = 'D:/CodeSpace/Python/ANN/files/training-package/verification/make/'
name = 'test.png'


def reload(path, name, toWidth=100, toHigh=40):
    image = IMP.resize(IMP.readGrayscale(path, name), toWidth=toWidth, toHigh=toHigh, isChannelPicture=True)
    IMP.showGrayscale(image)
    ret = np.argmax(MX_Prediction.prediction(np.array([[image]]), model), axis=1)
    lst = []
    for i in ret:
        lst.append(VG.choiceNum[int(i)])
    print(lst)


def prediction(toWidth=100, toHigh=40, codeSize=4, code: str = None):
    imageSize = (toWidth, toHigh)
    if code is None:
        code = VG.getCode(codeSize)
    else:
        code = code.upper()
    fonts = VG.getFont()
    fontsSize = len(fonts)

    item = fonts[random.randint(0, fontsSize - 1)]
    image = np.array(VG.generate(imageSize, code, pointSize=random.randint(0, 15), lineSize=random.randint(0, 10),
                                 fontSize=imageSize[1] * random.randint(3, 5) // 5, fontName=item))
    IMP.showPicture_RBG(image)
    image = IMP.channelsToGrayscale(image, False)
    IMP.showGrayscale(image)
    ret = np.argmax(MX_Prediction.prediction(np.array([[image]]), model), axis=1)
    lst = []
    for i in ret:
        lst.append(VG.choiceNum[int(i)])
    print(lst)


# reload(path, name, width, hight)
prediction(width, hight)
