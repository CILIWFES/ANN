import mxnet as mx


def get_ocrnet():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2, 2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5, 5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2, 2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3, 3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2, 2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

    flatten = mx.symbol.Flatten(data=relu3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512)
    fc21 = mx.symbol.FullyConnected(data=fc1, num_hidden=10)
    fc22 = mx.symbol.FullyConnected(data=fc1, num_hidden=10)
    fc23 = mx.symbol.FullyConnected(data=fc1, num_hidden=10)
    fc24 = mx.symbol.FullyConnected(data=fc1, num_hidden=10)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24], dim=0)
    label = mx.symbol.transpose(data=label)
    label = mx.symbol.Reshape(data=label, target_shape=(0,))
    return mx.symbol.SoftmaxOutput(data=fc2, label=label, name="softmax")

batch_size=31200
net=get_ocrnet()
# 输出参数情况供参考
shape = {"data": (batch_size, 3, 28, 28)}
mx.viz.print_summary(symbol=net, shape=shape)

# 由于训练数据多，这里采用了GPU，若读者没有GPU，可修改为CPU
module = mx.mod.Module(symbol=net, context=mx.gpu(0))
train_iter = mx.io.NDArrayIter(data=np.array(train_in), label={'reg_label': np.array(train_out)}, batch_size=batch_size,
                               shuffle=True)

# 训练
module.fit(
    train_iter,
    eval_data=val_iter,
    initializer=mx.init.MSRAPrelu(slope=0.0),  # 采用MSRAPrelu初始化
    optimizer='sgd',
    # 采用0.5的初始学习速率，并在每50000个样本后将学习速率缩减为之前的0.98倍
    optimizer_params={'learning_rate': 0.5,
                      'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=50000 / batch_size, factor=0.98)},
    num_epoch=200,
    batch_end_callback=mx.callback.Speedometer(batch_size, 50000 / batch_size),
    epoch_end_callback=mx.callback.do_checkpoint('D:/CodeSpace/Python/ANN/files/persistence/mxnet/test/simple')
)
