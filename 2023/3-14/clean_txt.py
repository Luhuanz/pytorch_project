import json
import re
import math
from opencc import OpenCC
import pandas as pd

cc = OpenCC('t2s')

country = ["丹麥", "日本", "荷蘭", "喀麦隆", "英格兰", "厄瓜多尔", "伊朗", "法國", "韩国", "墨西哥",  "葡萄牙", "塞内加尔", "加纳", "波蘭", "乌拉圭", "威尔士", "西班牙", "澳大利亚", "巴西", "瑞士", "加拿大", "摩洛哥", "阿根廷", "哥斯达黎加", "德國", "美國", "卡塔尔", "比利時", "克罗地亚", "沙烏地阿拉伯"]

newcountry = ['丹麦', '日本', '荷兰', '喀麦隆', '英格兰', '厄瓜多尔', '伊朗', '法国', '韩国', '墨西哥', '葡萄牙', '塞内加尔', '加纳', '波兰', '乌拉圭', '威尔士', '西班牙', '澳大利亚', '巴西', '瑞士', '加拿大', '摩洛哥', '阿根廷', '哥斯达黎加', '德国', '美国', '卡塔尔', '比利时', '克罗地亚', '沙乌地阿拉伯']

train_country = ["阿根廷", "澳大利亚", "巴西", "比利时", "波兰"]
dev_country = ["丹麦", "德国", "厄瓜多尔", "法国", "哥斯达黎加"]
#这段代码实现的是数据的清理。它会逐个读取 rawtxt/ 目录下的原始文本文件，去除括号内的内容，去除空格，
# 使用 OpenCC 库进行简繁体转换，并按行分割文本，然后将处理好的数据写入到 cleantxt/ 目录下的新文件中
def clean():
    for i,c in enumerate(country):
        with open("rawtxt/{}.txt".format(c), "r",encoding='utf-8') as f:
            data = f.read()
            a=r'\（.*?\）'
            data = re.sub(a, '', data) #从文本中删除括号和括号内的内容
            data = data.replace(' ','')#将字符串中的空格去掉
            data = cc.convert(data).split()
            newdata = []
            for d in data:
                if d=="":
                    continue
                else:
                    newdata.append(d+'\n')
        with open("cleantxt1/{}.txt".format(newcountry[i]), "w",encoding='utf-8') as f:
            f.writelines(newdata)

def write_lines(lines, path, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as f:
        for line in lines:
            f.writelines('{}\n'.format(line))

def get_data():
    train, dev, test = [], [], []
    for c in newcountry:
        with open("cleantxt/{}.txt".format(c), "r",encoding='utf-8') as f:
            data = f.read().split() # list
        if c in train_country+dev_country:
            with open("label/{}.txt".format(c), "r",encoding='utf-8') as f:
                label = f.read().split()
        else:
            label = []
            for d in data:
                label.append("0" * len(d))
        for i in range(len(data)):
            d = data[i]  # 每个字符对应 label的一个数字
            l = label[i]
            # print(len(d)==len(l))
            d = list(d)
            l = list(l)
            tmpl = []
            for j in l:  # ner编码 BIO no bioes
                if j=="1":
                    tmpl.append("B-NAME")
                elif j=="2":
                    tmpl.append("I-NAME")
                elif j=="3":
                    tmpl.append("B-ORG")
                elif j=="4":
                    tmpl.append("I-ORG")
                elif j=="5":
                    tmpl.append("B-LOC")
                elif j=="6":
                    tmpl.append("I-LOC")
                else:
                    tmpl.append("O")
            tmp = {}
            tmp["text"] = d
            tmp["label"] = tmpl
            #print(len(d)==len(tmpl))
    #，函数是将 Python 对象转换为 JSON 字符串的函数，设置 ensure_ascii=False 是为了保留中文字符，避免转义为 Unicode 转义序列。
            tmp = json.dumps(tmp, ensure_ascii=False)
            if c in train_country:
                train.append(tmp)
            if c in dev_country:
                dev.append(tmp)
            test.append(tmp)  #?
    write_lines(train, "nerdata/train.json")
    write_lines(dev, "nerdata/dev.json")
    write_lines(test, "nerdata/test.json")
# clean()
get_data()

# 0-
# 1-bname
# 2-iname
# 3-b-org
# 4-iorg
# 5-bloc
# 6-iloc