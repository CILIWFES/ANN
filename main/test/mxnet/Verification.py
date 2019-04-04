import mxnet as mx
import numpy as np
from main.dataprocessing import *
import logging

logging.getLogger().setLevel(logging.DEBUG)

print('加载图片')
(trainImages, trainLabels), (testImages, testLabels) = VG.loadSet()

batch_size = 50


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


def Branch(net, suffix, n_filter, stride=(1, 1)):
    # 回到主路
    net = ResBlock(net, 'restNet'+suffix, 1, n_filter, stride=stride)
    net = mx.sym.BatchNorm(net, name='bnBranch' + suffix, fix_gamma=False)
    net = mx.sym.Activation(net, name='actBranch' + suffix, act_type='relu')
    net = mx.sym.Pooling(net, name="pool" + suffix, pool_type="avg", kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    net = mx.sym.Flatten(net, name="flatten" + suffix)
    return net


label = mx.symbol.Variable('softmax_label')
net = mx.symbol.Variable('data')
net = mx.sym.Pooling(net, name="pool", pool_type="avg", kernel=(3, 3), stride=(1, 1), pad=(1, 1))
# # 为数据加上BN层可有一定的预处理效果
net = mx.sym.BatchNorm(net, name='bnPRE', fix_gamma=False)

# 将1*40*100变化为128*40*100
net = mx.sym.Convolution(net, name="convPRE", kernel=(3, 3), pad=(1, 1), num_filter=128)
# 将128*40*100变化为64*40*100
net = ResBlock(net, "1", 2, 64)
# 将64*40*100变化为64*20*50
net = ResBlock(net, "2", 2, 64, (2, 2))

# 将128*20*50变化为64*10*25
net1 = Branch(net, "1", 64, (2, 2))
# 将128*20*50变化为64*10*25
net2 = Branch(net, "2", 64, (2, 2))
# 将128*20*50变化为64*10*25
net3 = Branch(net, "3", 64, (2, 2))
# 将128*20*50变化为64*10*25
net4 = Branch(net, "4", 64, (2, 2))
# 将128变换为10

net1 = mx.symbol.FullyConnected(data=net1, num_hidden=36)
net2 = mx.symbol.FullyConnected(data=net2, num_hidden=36)
net3 = mx.symbol.FullyConnected(data=net3, num_hidden=36)
net4 = mx.symbol.FullyConnected(data=net4, num_hidden=36)
net = mx.symbol.Concat(*[net1, net2, net3, net4], dim=0)
label = mx.symbol.transpose(data=label)
label = mx.symbol.Reshape(data=label, target_shape=(0,))
net = mx.sym.SoftmaxOutput(net, name='Softmax', label=label)

print("加载网络")
# 输出参数情况供参考
# shape = {"data": (batch_size, 3, 20, 40)}
# mx.viz.print_summary(symbol=net, shape=shape)

# 由于训练数据多，这里采用了GPU，若读者没有GPU，可修改为CPU
module = mx.mod.Module(symbol=net, context=mx.gpu(0))
print('加载迭代器')
train_iter = mx.io.NDArrayIter(data=trainImages, label=trainLabels, batch_size=batch_size,
                               shuffle=True)
val_iter = mx.io.NDArrayIter(data=testImages, label=testLabels, batch_size=batch_size,
                             shuffle=True)
print("训练")

codeSize = 4


def Accuracy(label, pred, codeSize=4):
    label = label.T.reshape((-1,))
    pred_label = np.argmax(pred, axis=1)
    # hit = 0
    # length = len(pred_label) // codeSize
    # for i in range(len(pred_label) // codeSize):
    #     sum = 0
    #     for j in range(codeSize):
    #         index = j * length + i
    #         if pred_label[index] == label[index]:
    #             sum = sum + 1
    #         else:
    #             break
    #     if sum == codeSize:
    #         hit = hit + 1
    #
    # total = pred.shape[0] // codeSize
    # return 1.0 * hit / total
    hit = (pred_label == label).sum() // codeSize
    total = pred.shape[0] // codeSize
    return 1.0 * hit / total
    # hit = (pred_label == label).sum() // codeSize
    # total = pred.shape[0] // codeSize
    # return 1.0 * hit / total


# mx.metric.create('acc') 会运行 (pred_label == label).sum() 由于传入的label没有转置
# 会导致出现label=[1,2,1,2,1,2,1,2] 而pred_label 是[1,1,1,1,2,2,2,]这样子
# 2字识别极限为50%
sym, arg_params, aux_params = mx.model.load_checkpoint("D:/CodeSpace/Python/ANN/files/persistence/mxnet/verification/VG",
                                                       98)  # load with net name and epoch num

# 训练
module.fit(
    train_iter,
    arg_params=arg_params,
    aux_params=aux_params,
    begin_epoch=99,
    eval_data=val_iter,
    # initializer=mx.init.MSRAPrelu(slope=0.0),  # 采用MSRAPrelu初始化
    optimizer='sgd',
    eval_metric=Accuracy,
    # 采用0.1的初始学习速率，并在每5000个样本后将学习速率缩减为之前的0.98倍
    optimizer_params={'learning_rate': 0.08,
                      'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=10000 // batch_size, factor=0.9)},
    num_epoch=130,
    # batch_end_callback=mx.callback.Speedometer(batch_size, 50000 // batch_size),
    epoch_end_callback=mx.callback.do_checkpoint('D:/CodeSpace/Python/ANN/files/persistence/mxnet/test/VG')
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
#         kk = [np.argmax(module.get_outputs()[0],axis=1)]
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
