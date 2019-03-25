# 导入random模块
import random

# 导入Image,ImageDraw,ImageFont模块
from PIL import Image, ImageDraw, ImageFont
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
            img = IMP.readChannels(path, fileName)
            trainImages.append(img)
            label = fileName.split('_')[0]
            trainLabels.append(np.array([self.toNum(item) for item in label]))
        trainImages = np.array(trainImages)
        trainLabels = np.array(trainLabels)

        files = ORM.autoSearch(testPath)
        testImages = []
        testLabels = []
        for path, fileName in files:
            img = IMP.readChannels(path, fileName)
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
            img = self.generate(imageSize, code, pointSize=random.randint(5, 20), lineSize=random.randint(5, 20),
                                fontSize=imageSize[1] * random.randint(3, 5) // 5, fontName=item)
            self.save(img, code + '_' + item + '.png', trainPath)
        for i in range(testSize):
            code = self.getCode(codeSize)
            item = fonts[random.randint(0, fontSize - 1)]
            img = self.generate(imageSize, code, pointSize=random.randint(5, 20), lineSize=random.randint(5, 20),
                                fontSize=imageSize[1] * random.randint(3, 5) // 5, fontName=item)
            self.save(img, code + '_' + item + '.png', testPath)

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
    def generate(self, size: tuple, code, fontSize=None, fontName=None, lineSize=10, pointSize=10):
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
        self.generateText(draw, code, size, font)
        # if lineSize > 0:
        #     self.generateLine(draw, size[0], size[1], lineSize)
        # if pointSize > 0:
        #     self.generatePoint(draw, size[0], size[1], pointSize)
        return img

    # 填充文字
    def generateText(self, draw, code, size, font):
        x = size[0] // len(code)
        y = size[1]
        for num, item in enumerate(code):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.text((num * random.randint(x * 4 // 5, x), 0.1 * random.randint(-y, y)),
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
