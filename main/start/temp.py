# coding:utf-8
from urllib import request
import re


def regularization(content, match):
    pattern1 = re.compile(match)
    finds = pattern1.findall(content)
    return finds


######
# 爬虫v0.1 利用urlib 和 字符串内建函数
######
def getHtml(url):
    # 获取网页内容
    page = request.urlopen(url)
    html = page.read().decode('utf-8')
    return html


def content(html):
    match = '(?<=<article class="article-content">).*(?=</article>)'
    return regularization(html, match)[0]


def decode(content):
    match = r'(?<=<p>).+img.+?src.+?(?=</p>)'
    # match = r'(?<=<p>)[【[]\d*[】].*?(?=</p>)'
    lst = regularization(content, match)
    return lst


def get_img(content, beg=0):
    # 匹配图片的url
    # 思路是利用str.index()和序列的切片
    try:
        img_list = []
        while True:
            src1 = content.index('http', beg)
            src2 = content.index('/></p>', src1)
            img_list.append(content[src1:src2])
            beg = src2

    except ValueError:
        return img_list


def many_img(data, beg=0):
    # 用于匹配多图中的url
    try:
        many_img_str = ''
        while True:
            src1 = data.index('http', beg)
            src2 = data.index(' /><br /> <img src=', src1)
            many_img_str += data[src1:src2] + '|'  # 多个图片的url用"|"隔开
            beg = src2
    except ValueError:
        return many_img_str


def data_out(title, img):
    # 写入文本
    with open("/home/qq/data.txt", "a+") as fo:
        fo.write('\n')
        for size in range(0, len(title)):
            # 判断img[size]中存在的是不是一个url
            if len(img[size]) > 70:
                img[size] = many_img(img[size])  # 调用many_img()方法
            fo.write(title[size] + '$' + img[size] + '\n')


content = content(getHtml("https://bh.sb/post/43506/"))
match = """(?<=<p>).*?[【\[]\d*[】\]].+?src=["'].+?["'].+?(?=</p>)"""
# 或者
# match = """(?<=<p>).+?(?:</p>).*?(?:<p>).+?(?=</p>)"""
lst = regularization(content, match)
# title = decode(content)
# img = get_img(content)
# data_out(title, img)

xx = """
<body>
<h1>Welcome to my page</H1>
Content is divided into twosections:<br>
<h2>Introduction</h2>
Information about me.
<H2>Hobby</H2>
Information about my hobby.
<h2>This is invalid HTML</h3>
</body>
"""

match='''(?<=<)[hH]([1-6])>.*?</[hH]\\1>'''
lst = regularization(xx, match)