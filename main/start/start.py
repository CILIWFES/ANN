from main.configuration import *

import mxnet as mx
print("xx")
a = mx.nd.ones((2, 3), mx.gpu())
print(a)
print((mx.nd.ones((2, 2), mx.cpu()) * 100).asnumpy())
print((mx.nd.ones((2, 2), mx.gpu()) * 100).asnumpy())
