import mxnet as mx
from main.dataprocessing import *
import logging

logging.getLogger().setLevel(logging.DEBUG)

print('加载图片')
(trainImages, trainLabels), (testImages, testLabels) = VG.loadSet()


#
# def get_ocrnet():
#     data = mx.symbol.Variable('data')
#     label = mx.symbol.Variable('softmax_label')
#     bn = mx.sym.BatchNorm(data=data, name="bn", fix_gamma=False)
#     # 100x40x3
#     conv1 = mx.symbol.Convolution(data=bn, kernel=(3, 3), num_filter=96, pad=(1, 1), stride=(1, 1))
#     # 100x40x96
#     pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(4, 4), stride=(2, 2), pad=(1, 1))
#     # 50 x 20 96
#     relu1 = mx.symbol.Activation(data=pool1, act_type="relu")
#     # 50x20 x96
#     conv2 = mx.symbol.Convolution(data=relu1, kernel=(3, 3), num_filter=96, stride=(1, 1), pad=(1, 1))
#     # 50x20 x96
#     pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(4, 4), stride=(2, 2), pad=(1, 1))
#     # 25x10x96
#     relu2 = mx.symbol.Activation(data=pool2, act_type="relu")
#
#     # 25x20x64
#     conv3 = mx.symbol.Convolution(data=relu2, kernel=(3, 3), num_filter=32, pad=(1, 1), stride=(2, 2))
#     # 13x5x32
#     bn2 = mx.sym.BatchNorm(conv3, name="bn2", fix_gamma=False)
#     relu3 = mx.symbol.Activation(data=bn2, act_type="relu")
#
#     flatten = mx.symbol.Flatten(data=relu3)
#     fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=2080)
#     fc21 = mx.symbol.FullyConnected(data=fc1, num_hidden=36)
#     fc22 = mx.symbol.FullyConnected(data=fc1, num_hidden=36)
#     fc23 = mx.symbol.FullyConnected(data=fc1, num_hidden=36)
#     fc24 = mx.symbol.FullyConnected(data=fc1, num_hidden=36)
#     fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24], dim=0)
#     label = mx.symbol.transpose(data=label)
#     label = mx.symbol.Reshape(data=label, target_shape=(0,))
#     return mx.symbol.SoftmaxOutput(data=fc2, label=label, name="softmax")
#

# 残差模块
def ResBlock(net, suffix, n_block, n_filter, stride=(1, 1)):
    for i in range(0, n_block):
        if i == 0:  # 注意第1个残差层的定义不同，读者可观察结构图思考原因
            net = mx.sym.BatchNorm(net, name='bn' + suffix + '_a' + str(i), fix_gamma=False)
            net = mx.sym.Activation(net, name='act' + suffix + '_a' + str(i), act_type='relu')
            # 对于第1个残差层，旁路从此开始
            pathway = mx.sym.Convolution(net, name="adj" + suffix, kernel=(1, 1), stride=stride, num_filter=n_filter)
            # 回到主路
            net = mx.sym.Convolution(net, name='conv' + suffix + '_a' + str(i), kernel=(3, 3), pad=(1, 1),
                                     num_filter=n_filter, stride=stride)
            net = mx.sym.BatchNorm(net, name='bn' + suffix + '_b' + str(i), fix_gamma=False)
            net = mx.sym.Activation(net, name='act' + suffix + '_b' + str(i), act_type='relu')
            net = mx.sym.Convolution(net, name='conv' + suffix + '_b' + str(i), kernel=(3, 3), pad=(1, 1),
                                     num_filter=n_filter)
            net = net + pathway  # 加上旁路，即为残差结构
        else:
            pathway = net  # 对于后续残差层，旁路从此开始
            net = mx.sym.BatchNorm(net, name='bn' + suffix + '_a' + str(i), fix_gamma=False)
            net = mx.sym.Activation(net, name='act' + suffix + '_a' + str(i), act_type='relu')
            net = mx.sym.Convolution(net, name='conv' + suffix + '_a' + str(i), kernel=(3, 3), pad=(1, 1),
                                     num_filter=n_filter)
            net = mx.sym.BatchNorm(net, name='bn' + suffix + '_b' + str(i), fix_gamma=False)
            net = mx.sym.Activation(net, name='act' + suffix + '_b' + str(i), act_type='relu')
            net = mx.sym.Convolution(net, name='conv' + suffix + '_b' + str(i), kernel=(3, 3), pad=(1, 1),
                                     num_filter=n_filter)
            net = net + pathway  # 加上旁路，即为残差结构
    return net


label = mx.symbol.Variable('softmax_label')
net = mx.symbol.Variable('data')
# 为数据加上BN层可有一定的预处理效果
net = mx.sym.BatchNorm(net, name='bnPRE', fix_gamma=False)
# 将3*32*32变化为32*32*32
net = mx.sym.Convolution(net, name="convPRE", kernel=(3, 3), pad=(1, 1), num_filter=32)
# 将32*32*32变化为64*32*32
net = ResBlock(net, "1", 3, 64)
# 将64*32*32变化为64*16*16
net = ResBlock(net, "2", 3, 64, (2, 2))
# 将64*16*16变化为128*8*8
net = ResBlock(net, "3", 3, 128, (2, 2))
# 因为此前的最终层是卷积，因此再做BN和Act处理
net = mx.sym.BatchNorm(net, name='bnFINAL', fix_gamma=False)
net = mx.sym.Activation(net, name='actFINAL', act_type='relu')
# 将128*8*8变化为128*1*1
net = mx.sym.Pooling(net, name="pool", global_pool=True, pool_type="avg", kernel=(1, 1))
# 将128*1*1变换为128
net = mx.sym.Flatten(net, name="flatten")
# 将128变换为10

net = mx.sym.FullyConnected(net, num_hidden=12, name='fc')

net = mx.sym.LogisticRegressionOutput(net, name='log', label=label)

batch_size = 100
print("加载网络")
# 输出参数情况供参考
shape = {"data": (batch_size, 3, 20, 40)}
mx.viz.print_summary(symbol=net, shape=shape)

# 由于训练数据多，这里采用了GPU，若读者没有GPU，可修改为CPU
module = mx.mod.Module(symbol=net, context=mx.gpu(0))
print('加载迭代器')
train_iter = mx.io.NDArrayIter(data=trainImages, label=trainLabels, batch_size=batch_size,
                               shuffle=True)
print("训练")
# 训练
module.fit(
    train_iter,
    # eval_data=val_iter,
    initializer=mx.init.MSRAPrelu(slope=0.0),  # 采用MSRAPrelu初始化
    optimizer='sgd',
    eval_metric=mx.metric.create('mse'),
    # 采用0.5的初始学习速率，并在每50000个样本后将学习速率缩减为之前的0.98倍
    optimizer_params={'learning_rate': 0.0001,
                      'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=50000 / batch_size, factor=0.98)},
    num_epoch=200,
    batch_end_callback=mx.callback.Speedometer(batch_size, 50000 / batch_size),
    # epoch_end_callback=mx.callback.do_checkpoint('D:/CodeSpace/Python/ANN/files/persistence/mxnet/test/simple')
)

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
