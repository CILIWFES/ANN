# 导入random模块
import random

# 导入Image,ImageDraw,ImageFont模块
from PIL import Image, ImageDraw, ImageFont
from main.dataprocessing import *


class VerificationGenerate:
    def __init__(self, SYSFontPath=None, Ignore=None):
        if Ignore is None:
            self.Ignore = '.fon'
        if SYSFontPath is None:
            self.SYSFontPath = 'C:/Windows/Fonts/'
        self.Fonts = self.getFont()

    def getFont(self):
        fonts = []
        list = ORM.autoSearch(self.SYSFontPath)
        for item in list:
            if item[1].find(self.Ignore) == -1:
                fonts.append(item[1])
        return fonts

    def save(self, imgs: list, name: list, path, format="png"):
        for index, item in enumerate(imgs):
            with open(path + name[index], "wb") as f:
                item.save(f, format=format)

    def getCode(self, codeLength):
        code = ""
        for i in range(codeLength):
            code += random.choice([chr(random.randint(65, 90)), str(random.randint(0, 9))])
        return code

    def generate(self, size: tuple, code, amount=1, fontSize=None, fontName=None, lineSize=10):
        if fontSize is None:
            fontSize = min(size[0], size[1]) // 2
        imgs = list()
        for i in range(amount):
            # 定义使用Image类实例化一个长为120px,宽为30px,基于RGB的(255,255,255)颜色的图片
            img = Image.new(mode="RGB", size=size, color=(255, 255, 255))
            # 实例化一支画笔
            draw = ImageDraw.Draw(img, mode="RGB")
            if fontName is not None:
                font = ImageFont.truetype(fontName, fontSize)
            else:
                font = ImageFont.truetype(self.Fonts[random.randint(0, len(self.Fonts))], fontSize)

            for index, item in enumerate(code):
                # 宽度调整,高度无所谓
                self.generateText(draw, item, size[0]//len(code), size[1], index, font)
            for i in range(lineSize):
                self.generateLine(draw, size[0], size[1])
            imgs.append(img)
        return imgs

    def generateText(self, draw, code, width, height, num, font):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.text(
            (num * random.randint(width * 4 // 5, width), 0.1 * random.randint(-height, height)),
            code, color, font=font)

    def generateLine(self, draw, width, height):
        linecolor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        begin = (random.randint(0, width), random.randint(0, height))
        end = (random.randint(0, width), random.randint(0, height))
        draw.line([begin, end], fill=linecolor)

    def garble(self, img):
        return ''
