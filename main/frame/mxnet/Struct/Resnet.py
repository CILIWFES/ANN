import mxnet as mx


# 残差模块 suffix:序号 n_block:残差网路数量
# net=ResBlock(net, "1", 3, 64, (2, 2))
def ResBlock(net, suffix, n_block, n_filter, stride=(1, 1)):
    for i in range(n_block):
        pathway = net
        net = mx.sym.BatchNorm(net, name='bn' + suffix + '_a' + str(i), fix_gamma=False)
        net = mx.sym.Activation(net, name='act' + suffix + '_a' + str(i), act_type='relu')
        if i == 0:
            # 注意第1个残差层的定义不同，这里进行1x1 升/降维度
            # 防止上一层传入维度与输出冲突
            pathway = mx.sym.Convolution(net, name="adj" + suffix, kernel=(1, 1), stride=stride, num_filter=n_filter)

        # 对于后续残差层，旁路从此开始
        net = mx.sym.Convolution(net, name='conv' + suffix + '_a' + str(i), kernel=(3, 3), pad=(1, 1),
                                 num_filter=n_filter)
        net = mx.sym.BatchNorm(net, name='bn' + suffix + '_b' + str(i), fix_gamma=False)
        net = mx.sym.Activation(net, name='act' + suffix + '_b' + str(i), act_type='relu')
        net = mx.sym.Convolution(net, name='conv' + suffix + '_b' + str(i), kernel=(3, 3), pad=(1, 1),
                                 num_filter=n_filter)
        # 加上旁路，即为残差结构
        net = net + pathway
    return net
