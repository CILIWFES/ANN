# 导入random模块
import random

# 导入Image,ImageDraw,ImageFont模块
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from main.dataprocessing import *
from main.configuration import *
import numpy as np


class VerificationGenerate:
    fonts = []
    numChoice = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
        , 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19
        , 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29
        , 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35
    }
    choiceNum = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
        , 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J'
        , 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T'
        , 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
    }
    trainPath = GLOCF.getFilsPath(GLOCT.TRAINING_SECTION, [GLOCT.COMMON_CONFIG_FOLDER,
                                                           GLOCT.TRAINING_VERIFICATION_PATH,
                                                           GLOCT.TRAINING_VERIFICATION_TRAIN])

    testPath = GLOCF.getFilsPath(GLOCT.TRAINING_SECTION, [GLOCT.COMMON_CONFIG_FOLDER,
                                                          GLOCT.TRAINING_VERIFICATION_PATH,
                                                          GLOCT.TRAINING_VERIFICATION_TEST])

    # 初始化加载字体库
    def __init__(self, SYSFontPath=None, Ignore=None):
        # 过滤字体
        if Ignore is None:
            self.Ignore = ['.fon', '.dat', '.xml', '.ini', 'BSSYM7', 'FRSCRIPT', 'GIGI', 'HARLOWSI', 'HARNGTON',
                           'holomdl2', 'ITCBLKAD', 'ITCEDSCR', 'KUNSTLER', 'MAGNETOB', 'marlett', 'MATURASC', 'MISTRAL',
                           'MTCORSVA', 'MTEXTRA', 'OLDENGL', 'OUTLOOK', 'PALSCRI', 'PARCHM', 'PERTILI', 'POORICH',
                           'PRISTINA', 'RAGE', 'RAVIE', 'SCRIPTBL', 'segmdl2', 'SHOWG', 'SNAP____', 'STCAIYUN',
                           'STXINGKA', 'BRUSHSCI', 'symbol', 'VIVALDII', 'VLADIMIR', 'webdings', 'wingding', 'WINGDNG2',
                           'WINGDNG3', 'GLSNECB', 'GILSANUB', 'LATINWD', 'GOUDYSTO', 'JUICE___', 'REFSPCL']
        # 默认windows字体
        if SYSFontPath is None:
            self.SYSFontPath = 'C:/Windows/Fonts/'
        # 开始加载
        self.Fonts = self.getFont()

    # 是否包含判断,用于过滤字体
    def _contain(self, name):
        for item in self.Ignore:
            if name.find(item) != -1:
                return True
        return False

    # 加载数字集合
    def loadSet(self):
        trainPath = VerificationGenerate.trainPath
        testPath = VerificationGenerate.testPath
        files = ORM.autoSearch(trainPath)
        trainImages = []
        trainLabels = []
        for path, fileName in files:
            pil = IMP.readGrayscale(path, fileName)
            # pil = IMP.threshold(pil)
            # pil = IMP.noise_remove_pil(pil, 4)
            # ORM.savePicture(path,fileName,pil)
            img = np.array([pil])
            trainImages.append(img)
            label = fileName.split('_')[0]
            trainLabels.append(np.array([self.toNum(item) for item in label]))
        trainImages = np.array(trainImages)
        trainLabels = np.array(trainLabels)

        files = ORM.autoSearch(testPath)
        testImages = []
        testLabels = []
        for path, fileName in files:
            pil = IMP.readGrayscale(path, fileName)
            # pil = IMP.threshold(pil)
            # pil = IMP.noise_remove_pil(pil, 4)
            # ORM.savePicture(path,fileName,pil)
            img = np.array([pil])
            testImages.append(img)
            label = fileName.split('_')[0]
            testLabels.append(np.array([self.toNum(item) for item in label]))
        testImages = np.array(testImages)
        testLabels = np.array(testLabels)

        return (trainImages, trainLabels), (testImages, testLabels)

    # 0-9 A-Z 字符转化数字
    def toNum(self, char):
        numIndex = VerificationGenerate.numChoice[char]
        ret = numIndex
        return ret

    # 生成训练集(有线,点,偏移模式) VG.makeSet((100, 40), 31200, 3120)
    def makeSet(self, imageSize, trainSize, testSize, codeSize=4):
        trainPath = VerificationGenerate.trainPath
        testPath = VerificationGenerate.testPath

        fonts = self.getFont()
        fontSize = len(fonts)
        for i in range(trainSize):
            code = self.getCode(codeSize)
            item = fonts[random.randint(0, fontSize - 1)]
            img = self.generate(imageSize, code, pointSize=random.randint(0, 15), lineSize=random.randint(0, 15),
                                fontSize=imageSize[1] * random.randint(3, 5) // 5, fontName=item)

            self.save(img, code + '_' + item + '.png', trainPath)
        for i in range(testSize):
            code = self.getCode(codeSize)
            item = fonts[random.randint(0, fontSize - 1)]
            img = self.generate(imageSize, code, pointSize=random.randint(0, 15), lineSize=random.randint(0, 15),
                                fontSize=imageSize[1] * random.randint(3, 5) // 5, fontName=item)
            self.save(img, code + '_' + item + '.png', testPath)

    # 扭曲
    # 对当前图像进行透视变换，产生给定尺寸的新图像。
    # 变量data是一个8元组(a,b,c,d,e,f,g,h)，包括一个透视变换的系数。
    # 对于输出图像中的每个像素点
    # 新的值来自于输入图像的位置的(a x + b y + c)/(g x + h y + 1), (d x+ e y + f)/(g x + h y + 1)像素
    # 使用最接近的像素进行近似。
    def generateDistortion(self, img, imageSize, params=None):
        if params is None:
            # 图形扭曲参数
            params = [1,
                      random.randint(-5, 5) / 20,
                      0,
                      0,
                      1,
                      0,
                      0,
                      0
                      ]

        img = img.transform(imageSize, Image.AFFINE, params, fillcolor=(255, 255, 255))  # 创建扭曲

        # img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 滤镜，边界加强（阈值更大）
        return img

    # 加载字体库
    def getFont(self):
        fonts = []
        if len(VerificationGenerate.fonts) > 1:
            return VerificationGenerate.fonts
        list = ORM.autoSearch(self.SYSFontPath)
        for item in list:
            if not self._contain(item[1]):
                fonts.append(item[1])
        VerificationGenerate.fonts = fonts
        return fonts

    # 保存一张图片
    def save(self, img, name, path, format="png"):
        with open(path + name, "wb") as f:
            img.save(f, format=format)

    # 生成一定长度的文字
    def getCode(self, codeLength):
        code = ""
        for i in range(codeLength):
            code += random.choice([chr(random.randint(65, 90)), str(random.randint(0, 9))])
        return code

    # 生成一个验证码,文字为code,size=(x,y)
    def generate(self, size: tuple, code, fontSize=None, fontName=None, lineSize=10, pointSize=10, distortion=True):
        if fontSize is None:
            fontSize = size[1] * 4 // 5
        if fontName is None:
            fontName = self.Fonts[random.randint(0, len(self.Fonts) - 1)]
        # 定义使用Image类实例化一个长为120px,宽为30px,基于RGB的(255,255,255)颜色的图片
        img = Image.new(mode="RGB", size=size, color=(255, 255, 255))
        # 实例化一支画笔
        draw = ImageDraw.Draw(img, mode="RGB")
        font = ImageFont.truetype(fontName, fontSize)
        # 填充具有上下左右偏移的文字
        self.generateText(draw, code, size, font, pad=8)
        if distortion:
            img = self.generateDistortion(img, size)
            draw = ImageDraw.Draw(img, mode="RGB")
        if lineSize > 0:
            self.generateLine(draw, size[0], size[1], lineSize)
        if pointSize > 0:
            self.generatePoint(draw, size[0], size[1], pointSize)
        return img

    def generateChannels(self, size: tuple, code, fontSize=None, fontName=None, lineSize=10, pointSize=10,
                         distortion=True):
        return IMP.conversionChannels(
            np.array(self.generate(size, code, fontSize, fontName, lineSize, pointSize, distortion)))

    def generateGrayscale(self, size: tuple, code, fontSize=None, fontName=None, lineSize=10, pointSize=10,
                          distortion=True):
        return IMP.channelsToGrayscale(
            np.array(self.generate(size, code, fontSize, fontName, lineSize, pointSize, distortion)), False)

    # 填充文字
    def generateText(self, draw, code, size, font, offsetMax=0.2, pad=5):
        x = (size[0] - 2 * pad) // len(code)
        y = size[1]
        for num, item in enumerate(code):
            ran = random.randint(-int(x * 0.5 * offsetMax), int(x * offsetMax))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.text((num * x + ran + pad, offsetMax * random.randint(-y, y)),
                      item, color, font=font)

    # 填充点
    def generatePoint(self, draw, width, height, pointSize):
        for i in range(pointSize):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.point([random.randint(0, width), random.randint(0, height)], fill=color)
            x = random.randint(0, width)
            y = random.randint(0, height)
            draw.arc((x, y, x + 4, y + 4), 0, 90, fill=color)

    # 填充线
    def generateLine(self, draw, width, height, lineSize):
        for i in range(lineSize):
            linecolor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            begin = (random.randint(0, width), random.randint(0, height))
            end = (random.randint(0, width), random.randint(0, height))
            draw.line([begin, end], fill=linecolor)
