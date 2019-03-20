import pickle
import os
import cv2


class ORM:
    # 写入对象的序列化
    def writePickle(self, path, fileName, object):
        if not os.path.exists(path):
            os.makedirs(path)
        fileObj = open(path + fileName, "wb")
        pickle.dump(object, fileObj)
        fileObj.close()

    # 读取对象的序列化
    def LoadPickle(self, path):
        fileObj = open(path, "rb")
        object = pickle.load(fileObj)
        fileObj.close()
        return object

    # 保存文件
    def saveFile(self, savePath, fileName, content):
        if not os.path.exists(savePath):
            os.makedirs(savePath)  # 若不存在则创建目录
        content = content.encode(encoding='utf-8')  # 解码为字节码
        fp = open(savePath + fileName, "wb")
        fp.write(content)
        fp.close()

    # 保存图片
    def savePicture(self, savePath, fileName, image):
        if not os.path.exists(savePath):
            os.makedirs(savePath)  # 若不存在则创建目录
        cv2.imwrite(savePath + fileName, image)

    # 读取文件
    def readFile(self, classPath, fileName):
        fp = open(classPath + fileName, "rb")
        content = fp.read()
        content = content.decode(encoding='utf-8').strip()  # 解码为字符码
        fp.close()
        return content
        # 读取文件

    #  嗅探文件,返回[(路径,名字)]
    # seachPath="xxx/ss/"
    def autoSearch(self, seachPath):
        fileinfo = []
        cateList = os.listdir(seachPath)
        for mydir in cateList:
            if os.path.isfile(seachPath+mydir):
                fileinfo.append((seachPath, mydir))
            else:
                classPath = seachPath + mydir + '/'
                temp = self.autoSearch(classPath)
                fileinfo.extend(temp)

        return fileinfo

    # 判断文件/文件夹是否存在
    def exist(self, path):
        if os.path.exists(path):
            return True
        else:
            return False
