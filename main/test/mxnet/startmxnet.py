from main.configuration import *
import logging
import math
import random
import mxnet as mx
import numpy as np

cup_gpu = mx.cpu()

logging.getLogger().setLevel(logging.DEBUG)
print("开始")

n_sample = 10000
# 一批内的数量,越小对逻辑运算要求越高
batch_size = 100
learning_rate = 0.1
n_epoch = 10

train_in = [[random.uniform(0, 1) for c in range(2)] for n in range(n_sample)]
train_out = [0 for n in range(n_sample)]
for i in range(n_sample):
    train_out[i] = max(train_in[i][0], train_in[i][1])

train_iter = mx.io.NDArrayIter(data=np.array(train_in), label={'reg_label': np.array(train_out)}, batch_size=batch_size,
                               shuffle=True)

src = mx.sym.Variable('data')

fc1 = mx.sym.FullyConnected(data=src, num_hidden=10, name='fc1')
act1 = mx.sym.Activation(data=fc1, act_type='relu', name='act1')

fc2 = mx.sym.FullyConnected(data=act1, num_hidden=1, name='fc3')

net = mx.sym.LinearRegressionOutput(data=fc2, name='reg')

module = mx.mod.Module(symbol=net, label_names=({'reg_label'}), context=cup_gpu)


def epoch_calback(epoch, symbol, arg_params, sux_params):
    for k in arg_params:
        print(k)
        print(arg_params[k].asnumpy())


module.fit(
    train_iter
    , eval_data=None
    , eval_metric=mx.metric.create('mse')
    , optimizer='sgd'
    , optimizer_params={'learning_rate': learning_rate}
    , num_epoch=n_epoch
    , initializer=mx.initializer.Uniform(0.5)
    , batch_end_callback=mx.callback.Speedometer(batch_size, 10)
    # , epoch_end_callback=epoch_calback
    # , batch_end_callback=None
    , epoch_end_callback=None
)

# pre_iter = mx.io.NDArrayIter(data=np.array(train_in[0:10]))
# y = module.predict(pre_iter)
# print(y)
