# from main.test.mxnet.resnet import *
import random
from main.dataprocessing import *

imgs = VG.generate((250, 50), VG.getCode(6), fontSize=50, amount=1)


VG.save(imgs, ['xx.png'], 'D:/CodeSpace/Python/ANN/files/temp/')
