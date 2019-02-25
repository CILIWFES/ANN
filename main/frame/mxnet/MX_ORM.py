import mxnet as mx
from main.configuration import *


class MX_ORM:
    # mxnet 保存module
    # Mx_ORM.Module_Save(module, "文件名字", 当前epoch次数(20))
    def Module_Save(self, module, fileName, epoch, save_optimizer_states=False, path=None):
        # 保存路径为空默认填充
        if path is None:
            path = GLOCF.getFilsPath(GLOCT.PERSISTENCE_SECTION,
                                     [GLOCT.COMMON_CONFIG_FOLDER, GLOCT.PERSISTENCE_MXNET_FOLDER])
        # 加载保存点
        module.save_checkpoint(path + fileName, epoch, save_optimizer_states)

    # 读取保存的mxnet module信息
    # 如 :Mx_ORM.Module_Read("test", 20, (batch_size, 1, test_rows, test_cols), (batch_size,))
    def Module_Read(self, fileName, epoch, dataStruct, labelStruct, dataName='data', labelName='softmax_label',
                    path=None,
                    cpuType=False):
        # 判断硬件类型
        processor = mx.cpu() if (cpuType) else mx.gpu()
        # 读取路径为空默认填充
        if path is None:
            path = GLOCF.getFilsPath(GLOCT.PERSISTENCE_SECTION,
                                     [GLOCT.COMMON_CONFIG_FOLDER, GLOCT.PERSISTENCE_MXNET_FOLDER])
        # 读取数据
        sym, arg_params, aux_params = mx.model.load_checkpoint(path + fileName, epoch)

        # 加载module
        mod = mx.mod.Module(symbol=sym, context=processor, data_names=[dataName], label_names=[labelName])
        mod.bind(for_training=False, data_shapes=[(dataName, dataStruct)], label_shapes=[(labelName, labelStruct)])
        mod.set_params(arg_params, aux_params)
        return mod
