# from main.test.mxnet.Verification import *

import re

# key = "javapythonhtmlvhdl"#这是源文本
# p1 = "python"#这是我们写的正则表达式
# pattern1 = re.compile(p1)  # 我们在编译这段正则表达式
# matcher1 = re.search(pattern1, key)  # 在源文本中搜索符合正则表达式的部分
# print(matcher1.group(0))  # 打印出来

key = r"<h1>hello world</h1><h1>435343345</h1>"  # 源文本
p1 = r"<h1>.+?</h1>"  # 我们写的正则表达式，下面会将为什么
pattern1 = re.compile(p1)
print(pattern1.findall(key))
