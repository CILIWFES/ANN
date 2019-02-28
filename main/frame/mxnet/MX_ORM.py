import mxnet as mx
import os
from main.configuration import *


class MX_ORM:
    # mxnet 保存module
    # Mx_ORM.Module_Save(module, "文件名字", 当前epoch次数(20))
    def Module_Save(self, module, fileName, path=None, epoch=None, save_optimizer_states=False):
        # 保存路径为空默认填充
        if path is None:
            path = GLOCF.getFilsPath(GLOCT.PERSISTENCE_SECTION,
                                     [GLOCT.COMMON_CONFIG_FOLDER, GLOCT.PERSISTENCE_MXNET_FOLDER])

        if not os.path.exists(path):
            os.makedirs(path)  # 若不存在则创建目录
        prefix = path + fileName
        # 保存网络结构
        module._symbol.save(prefix + '-symbol.json')
        if epoch:
            param_name = '%s-%04d.params' % (prefix, epoch)
        else:
            param_name = prefix + ".params"

        # 保存参数
        module.save_params(param_name)

    # 读取保存的mxnet module信息
    # 如 :Mx_ORM.Module_Read("test", (batch_size, 1, test_rows, test_cols))
    def Module_Read(self, fileName, dataStruct, batch_size=None, epoch=None, dataName=None, labelName=None, path=None,
                    cpuType=False):
        # 判断硬件类型
        processor = mx.cpu() if (cpuType) else mx.gpu()
        # 读取路径为空默认填充
        if path is None:
            path = GLOCF.getFilsPath(GLOCT.PERSISTENCE_SECTION,
                                     [GLOCT.COMMON_CONFIG_FOLDER, GLOCT.PERSISTENCE_MXNET_FOLDER])
        prefix = path + fileName

        if epoch:
            param_name = '%s-%04d.params' % (prefix, epoch)
        else:
            param_name = prefix + ".params"

        symbol = mx.sym.load(prefix + '-symbol.json')

        if dataName is None:
            lst = symbol.list_arguments()
            dataName = lst[0]

        if labelName is None:
            lst = symbol.list_arguments()
            labelName = lst[-1]

        save_dict = mx.nd.load(param_name)
        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
            if tp == 'aux':
                aux_params[name] = v

        # 加载module
        mod = mx.mod.Module(symbol=symbol, context=processor, data_names=[dataName], label_names=[labelName])

        if batch_size is None:
            batch_size = dataStruct[0]
        mod.bind(for_training=False, data_shapes=[(dataName, dataStruct)], label_shapes=[(labelName, (batch_size,))])
        mod.set_params(arg_params, aux_params)
        return mod
