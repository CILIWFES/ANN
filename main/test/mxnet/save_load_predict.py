from main.analysis import PerM, MPoint
from main.dataprocessing import FsMnist
from main.frame.mxnet import MX_ORM, MX_Prediction
import mxnet as mx
import logging as log

log.getLogger().setLevel(log.DEBUG)


def train_save(fileName, train_img, test_img, train_lbl, test_lbl, batch_size, train_rows, train_cols, path="",
               num_epoch=20, context=mx.gpu()):
    # 迭代器
    train_iter = mx.io.NDArrayIter(train_img, train_lbl, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(test_img, test_lbl, batch_size)

    # 设置网络
    data = mx.symbol.Variable('data')

    # 将图像摊平，例如1*28*28的图像就变为784个数据点，这样才可与普通神经元连接
    flatten = mx.sym.Flatten(data=data, name="flatten")

    # 第1层网络及非线性激活
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=128, name="fc1")
    act1 = mx.sym.Activation(data=fc1, act_type="relu", name="act1")

    # 第2层网络及非线性激活
    fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64, name="fc2")
    act2 = mx.sym.Activation(data=fc2, act_type="relu", name="act2")

    # 输出神经元，因为需分为10类，所以有10个神经元
    fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10, name="fc3")
    # SoftMax
    net = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

    shape = {"data": (batch_size, 1, train_rows, train_cols)}
    # 信息打印
    # mx.viz.print_summary(symbol=net, shape=shape)

    # 由于训练数据量较大，这里采用了GPU，若没有GPU，可修改为CPU
    module = mx.mod.Module(symbol=net, context=context)

    # 绘制结构图
    mx.viz.plot_network(symbol=net, shape=shape).view()

    module.fit(
        train_iter,
        eval_data=val_iter,
        optimizer='sgd',
        # 准确率
        eval_metric=mx.metric.create('acc'),
        # 采用0.2的初始学习速率，并在每60000个样本后（即每1个epoch后）将学习速率缩减为之前的0.9倍
        optimizer_params={'learning_rate': 0.3,
                          'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=60000 / batch_size, factor=0.9)},
        num_epoch=num_epoch,
        batch_end_callback=mx.callback.Speedometer(batch_size, 60000 / batch_size)
    )
    # 开始保存
    MX_ORM.Module_Save(module, path + fileName)


def prediction(test_img, test_lbl, batch_size=1, num_epoch=None, path="", fileName=None, module=None):
    if module is None:
        assert fileName is not None
        module = MX_ORM.Module_Read(path + fileName, (batch_size, 1, test_rows, test_cols), epoch=num_epoch)

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


(train_img, train_lbl, train_rows, train_cols) = FsMnist.read_TrainImg()
(test_img, test_lbl, test_rows, test_cols) = FsMnist.read_TestImg()
# 批大小
batch_size = 64
# train_save('test, train_img, test_img, train_lbl, test_lbl, batch_size, train_rows, train_cols, num_epoch=2)
prediction(test_img[10:30], test_lbl[10:30], 1, fileName="test")
