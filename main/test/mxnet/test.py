import numpy as np
import logging
import mxnet as mx
from main.dataprocessing import *

logging.getLogger().setLevel(logging.DEBUG)

batch_size = 100

(trainImages, trainLabels), (testImages, testLabels) = VG.loadSet()

data = mx.symbol.Variable('data')
label = mx.symbol.Variable('softmax_label')
conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=32)
pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2, 2), stride=(1, 1))
relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

conv2 = mx.symbol.Convolution(data=relu1, kernel=(5, 5), num_filter=32)
pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2, 2), stride=(1, 1))
relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=32)
pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

conv4 = mx.symbol.Convolution(data=relu3, kernel=(3,3), num_filter=32)
pool4 = mx.symbol.Pooling(data=conv4, pool_type="avg", kernel=(2,2), stride=(1, 1))
relu4 = mx.symbol.Activation(data=pool4, act_type="relu")

flatten = mx.symbol.Flatten(data=relu2)
net = mx.symbol.FullyConnected(data=flatten, num_hidden=21)
net = mx.symbol.SoftmaxOutput(data=net, label=label, name="softmax")



train_iter = mx.io.NDArrayIter(data=trainImages, label=trainLabels, batch_size=batch_size,
                               shuffle=True)
val_iter = mx.io.NDArrayIter(data=testImages, label=testLabels, batch_size=batch_size,
                             shuffle=True)
# 训练
module.fit(
    train_iter,
    eval_data=val_iter,
    initializer=mx.init.MSRAPrelu(slope=0.0),  # 采用MSRAPrelu初始化
    optimizer='sgd',
    # 采用0.5的初始学习速率，并在每50000个样本后将学习速率缩减为之前的0.984倍
    optimizer_params={'learning_rate': 0.5,
                      'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=50000 / batch_size, factor=0.984)},
    num_epoch=2000,
    batch_end_callback=mx.callback.Speedometer(batch_size, 50000 / batch_size)
)
#
# metric = mx.metric.create('acc')
# # 为测试，采用纯手动初始化：
# print('初始数据')
# module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
#
# print('初始参数')
# module.init_params(initializer=mx.init.MSRAPrelu(slope=0.0))
# print('初始优化器')
# module.init_optimizer()
# for epoch in range(1000):  # 手动训练，这里只训练1个epoch
#     train_iter.reset()  # 每个epoch需手动将迭代器复位
#     # 实际训练时，应在此调用 metric.reset() 将性能指标复位
#     for batch in train_iter:  # 对于每个batch...
#         # print('============ input ============')
#         # print(batch.data)  # 数据
#         # print(batch.label)  # 数据的标签
#         module.forward(batch, is_train=True)  # 前向传播
#         # print('============ output ============')
#         # print(module.get_outputs())  # 网络的输出
#         metric.reset()  # 这里希望看网络的训练细节，所以对于每个样本都将指标复位
#         module.update_metric(metric, batch.label)  # 更新指标
#         print('============ metric ============')
#         print(metric.get())  # 指标的情况
#         # module.backward()  # 反向传播，计算梯度
#         # print('============ grads ============')
#         # print(module._exec_group.grad_arrays)  # 输出梯度情况
#         module.update()  # 根据梯度情况，由优化器更新网络参数
#         # print('============ params ============')
#         # print(module.get_params())  # 输出新的参数
#         print('\n')
