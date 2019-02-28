import mxnet as mx
from main.dataprocessing import *
import logging
from main.analysis.performanceMeasure import *
from main.frame.mxnet import *

logging.getLogger().setLevel(logging.DEBUG)

(train_img, train_lbl, train_rows, train_cols) = FsMnist.read_TrainImg()
(test_img, test_lbl, test_rows, test_cols) = FsMnist.read_TestImg()
batch_size = 64  # 批大小
#
# # 迭代器
# train_iter = mx.io.NDArrayIter(train_img, train_lbl, batch_size, shuffle=True)
# val_iter = mx.io.NDArrayIter(test_img, test_lbl, batch_size)
#
# data = mx.symbol.Variable("data")
# # 卷积层
# # 32 个5x5过滤器
# conv1 = mx.sym.Convolution(data=data, name='conv1', kernel=(5, 5), num_filter=32)
# # bn层
# bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False)
# # 激活函数
# act1 = mx.sym.Activation(data=bn1, name='act1', act_type='relu')
#
# # 以3x3 步长为2进行池化
# pool1 = mx.sym.Pooling(data=act1, name='pool1', pool_type='max', kernel=(3, 3), stride=(2, 2))
#
# # 卷积层
# # 32 个5x5过滤器
# conv2 = mx.sym.Convolution(data=pool1, name='conv2', kernel=(5, 5), num_filter=64)
# # bn层
# bn2 = mx.sym.BatchNorm(data=conv2, name='bn2', fix_gamma=False)
# # 激活函数
# act2 = mx.sym.Activation(data=bn2, name='act2', act_type='relu')
#
# # 以3x3 步长为2进行池化
# pool2 = mx.sym.Pooling(data=act2, name='pool2', pool_type='max', kernel=(3, 3), stride=(2, 2))
#
# # 卷积层
# # 32 个5x5过滤器
# conv3 = mx.sym.Convolution(data=pool2, name='conv3', kernel=(3, 3), num_filter=10)
#
# # 全局平均池化
# pool3 = mx.sym.Pooling(data=conv3, name='pool3', global_pool=True, kernel=(1, 1), pool_type='avg')
#
# flattem = mx.sym.Flatten(data=pool3, name="flatten")
#
# net = mx.sym.SoftmaxOutput(data=flattem, name='softmax')
#
# shape = {'data': (batch_size, 1, train_rows, train_cols)}
# # mx.viz.plot_network(symbol=net, shape=shape).view()
# mx.viz.print_summary(symbol=net, shape=shape)
#
# module = mx.module.Module(symbol=net, context=mx.gpu())
#
# module.fit(
#     train_iter
#     , eval_data=val_iter
#     , optimizer='sgd'
#     , optimizer_params={'learning_rate': 0.03,
#                         'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=60000 / batch_size, factor=0.9)}
#     , num_epoch=20
#     , batch_end_callback=mx.callback.Speedometer(batch_size, 60000 / batch_size)
# )
# MX_ORM.Module_Save(module, 'conTest')


module = MX_ORM.Module_Read('conTest', (1, 1, test_rows, test_cols))
test_img = test_img[200:220]
test_lbl = test_lbl[200:220]
MPoint.setPoint()
ret = MX_Prediction.prediction(test_img, module)
MPoint.showPoint("预测完毕")

# 每行找最大
preClass = ret.argmax(axis=1)
print("预测值为:", preClass.tolist())
perM = PerM()
perM.fit(preClass.tolist(), test_lbl.tolist())
perM.printPRFData()
# 图像显示
FsMnist.showImg(test_img, test_lbl)