import mxnet as mx


def Alexnet(net, suffix):
    net = mx.sym.Convolution(name=suffix + 'conv1', data=net, kernel=(11, 11), stride=(4, 4), num_filter=96)
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Pooling(data=net, pool_type="max", kernel=(3, 3), stride=(2, 2))
    net = mx.sym.Convolution(name=suffix + 'conv2', data=net, kernel=(5, 5), pad=(2, 2), num_filter=256)
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Pooling(data=net, kernel=(3, 3), stride=(2, 2), pool_type="max")
    net = mx.sym.Convolution(name=suffix + 'conv3', data=net, kernel=(3, 3), pad=(1, 1), num_filter=384)
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Convolution(name=suffix + 'conv4', data=net, kernel=(3, 3), pad=(1, 1), num_filter=384)
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Convolution(name=suffix + 'conv5', data=net, kernel=(3, 3), pad=(1, 1), num_filter=256)
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Pooling(data=net, kernel=(3, 3), stride=(2, 2), pool_type="max")
    net = mx.sym.Flatten(data=net)
    net = mx.sym.FullyConnected(name=suffix + 'fc1', data=net, num_hidden=4096)
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Dropout(data=net, p=0.5)
    net = mx.sym.FullyConnected(name=suffix + 'fc2', data=net, num_hidden=4096)
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Dropout(data=net, p=0.5)
    net = mx.sym.FullyConnected(name=suffix + 'fc3', data=net, num_hidden=1000)
    net = mx.sym.SoftmaxOutput(data=net, name=suffix + 'softmax')
    return net
