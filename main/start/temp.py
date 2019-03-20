# 导入random模块
import random

# 导入Image,ImageDraw,ImageFont模块
from PIL import Image, ImageDraw, ImageFont
from main.dataprocessing import *

# 定义使用Image类实例化一个长为120px,宽为30px,基于RGB的(255,255,255)颜色的图片
img = Image.new(mode="RGB", size=(120, 30), color=(255, 255, 255))
list = ORM.autoSearch('C:/Windows/Fonts/')
print(len(list))
lis = []
ll=[]
for item in list:
    name = item[1]
    # 实例化一支画笔
    draw = ImageDraw.Draw(img, mode="RGB")
    # 定义要使用的字体
    try:
        font = ImageFont.truetype(name, 28)
    except Exception:
        lis.append(name)
        continue

    for i in range(5):
        # 每循环一次,从a到z中随机生成一个字母或数字
        # 65到90为字母的ASCII码,使用chr把生成的ASCII码转换成字符
        # str把生成的数字转换成字符串
        char1 = random.choice([chr(random.randint(65, 90)), str(random.randint(0, 9))])

        # 每循环一次重新生成随机颜色
        color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # 把生成的字母或数字添加到图片上
        # 图片长度为120px,要生成5个数字或字母则每添加一个,其位置就要向后移动24px
        draw.text([i * 24, 0], char1, color1, font=font)
    # 把生成的图片保存为"pic.png"格式
    with open("pic.png", "wb") as f:
        img.save(f, format="png")
print(lis)
print(len(lis))
print(len(ll))
#用来绘制干扰线
#
# def gene_line(draw,width,height):
#
#     begin = (random.randint(0, width), random.randint(0, height))
#
#     end = (random.randint(0, width), random.randint(0, height))
#
#     draw.line([begin, end], fill = linecolor)
#
# # 最后创建扭曲，加上滤镜，用来增强验证码的效果。
#
#
# image = image.transform((width+20,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)  #创建扭曲
#
# image = image.filter(ImageFilter.EDGE_ENHANCE_MORE) #滤镜，边界加强