from main.dataprocessing import *

(img, label, rows, cols) = FsMnist.read_TrainImg()
FsMnist.showImg(img[455:470], label[455:470])
