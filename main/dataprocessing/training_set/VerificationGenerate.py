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

    def generate(self, size: tuple, code, amount=1, fontSize=50, fontName=None):
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
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw.text([index * random.randint(fontSize // 2, fontSize), 0.1 * random.randint(-fontSize, fontSize)],
                          item, color, font=font)
            imgs.append(img)
        return imgs

    def garble(self, img):
        return ''
