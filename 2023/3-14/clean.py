import json
import re
import math
from opencc import OpenCC
import pandas as pd

cc = OpenCC('t2s') #其配置为 't2s'。其中，'t2s' 表示将繁体中文转换为简体中文。

country = ["丹麥", "日本", "荷蘭", "喀麦隆", "英格兰", "厄瓜多尔", "伊朗", "法國", "韩国", "墨西哥",  "葡萄牙", "塞内加尔", "加纳", "波蘭", "乌拉圭", "威尔士", "西班牙", "澳大利亚", "巴西", "瑞士", "加拿大", "摩洛哥", "阿根廷", "哥斯达黎加", "德國", "美國", "卡塔尔", "比利時", "克罗地亚", "沙烏地阿拉伯"]

newcountry = ['丹麦', '日本', '荷兰', '喀麦隆', '英格兰', '厄瓜多尔', '伊朗', '法国', '韩国', '墨西哥', '葡萄牙', '塞内加尔', '加纳', '波兰', '乌拉圭', '威尔士', '西班牙', '澳大利亚', '巴西', '瑞士', '加拿大', '摩洛哥', '阿根廷', '哥斯达黎加', '德国', '美国', '卡塔尔', '比利时', '克罗地亚', '沙乌地阿拉伯']

def convert(alldata): # alldata:list of dist #alldata某俱乐部成员信息
    final = []
    for data in alldata: # 每个球员
        tmp = {}
        for k in data:  # 遍历的是字典的键
            v = data[k] #字典的值
            k = cc.convert(k) #转化为简体中文
            #于查找一个子字符串在另一个字符串中的位置。
    #发现字典的一个键（即 k 变量）包含子字符串 "号码"，那么将该键修改为 "number"。接下来，该代码段对值（即 v 变量）进行处理。
            if k.find("号码") != -1:
                k = "number"
    #检查变量 v 是否为字符串类型。如果是，则执行接下来的代码块；否则跳过这个代码块。
                if isinstance(v, str):
                    v = cc.convert(v)
#使用字符串 "--" 表示缺失值。
                    if v == "--":
                        v = -1
                    else:
                    #从字符串 v 中提取数字
                        v = re.findall(r"\d+\.?\d*", v)[0]
                elif math.isnan(v):
                    v = -1
                v = int(v)
                tmp[k] = v
                continue
            elif k.find("位置") != -1: #如果字符串 k 中包含子字符串 "位置"，则执行接下来的代码块。
                k = "position"
            elif k.find("名字") != -1 or k.find("姓名") != -1:
                if not isinstance(v, str): #这行代码的作用是，如果该字段的值不是字符串类型，则跳过该字段。
                    break
                k = "name"
                idx1 = v.find("（")
                idx2 = v.find("(")
                if idx1!=-1:  #如果字符串 v 中包含中文或英文括号，则提取括号前面的内容。
                    v = v[:idx1]
                elif idx2!=-1:
                    v = v[:idx2]
            elif k.find("日期") != -1:
                k = "age"
                if isinstance(v, str): #如果该字段的值是字符串类型，则将其转换为简体中文，并提取其中的数字。
                    v = cc.convert(v)
                    v = re.findall(r"\d+\.?\d*", v)[0]
                elif math.isnan(v):
                    v = -1
                v = int(v)
            elif k.find("出") != -1:
                k = "apperance"
                if isinstance(v, str):
                    v = cc.convert(v)
                    v = re.findall(r"\d+\.?\d*", v)[0]
                elif math.isnan(v):
                    v = -1
                v = int(v)
            elif k.find("入球") != -1 or k.find("进球") != -1:
                k = "goal"
                if isinstance(v, str):
                    v = cc.convert(v)
                    v = re.findall(r"\d+\.?\d*", v)[0] # 提取数字
                elif math.isnan(v):
                    v = -1
                v = int(v)
            elif k.find("球队") != -1 or k.find("球会") != -1:
                k = "club"
                if isinstance(v,str):
                    idx1 = v.find("（")
                    idx2 = v.find("(")
                    if idx1!=-1:
                        v = v[:idx1]
                    elif idx2!=-1:
                        v = v[:idx2]
                else:
                    v = "none"
#在 if 语句块中处理完所有满足条件的属性之后，如果还有其它属性没有处理，则直接跳过这些属性的处理，继续执行下一次循环。
            else:
                continue
            if isinstance(v, str):
                v = cc.convert(v)
            elif isinstance(v, list):
                v = convert(v)
            tmp[k] = v
#如果字典 tmp 中不存在键值为 "name" 或 "age" 的元素，则该 continue 语句会跳过这个球员数据的处理，继续执行下一个循环，处理下一个球员的数据。
        if "name" not in tmp.keys() or "age" not in tmp.keys():
            continue
        final.append(tmp)
    return final

total = []
for c in country: # 传统国家列表
    with open('rawdata/{}.json'.format(c), 'r', encoding='utf-8') as file:
        alldata = json.load(file) # encoding那边会转成utf-8
    #print(alldata[0].keys())
    newdata = convert(alldata)
    # with open('../newdata/{}.json'.format(cc.convert(c)), 'w') as file:
    #     json.dump(newdata, file)
    for data in newdata: # 加一个键值对
        data["country"] = cc.convert(c)
    total += newdata

club = []
position = []
countrys = []
relation = []

for d in total:
    if d["club"] not in club:
        club.append(d["club"])
    if d["position"] not in position:
        position.append(d["position"])
    if d["country"] not in countrys:
        countrys.append(d["country"])
    relation.append([d["name"], d["club"], "work_for"]) # 实体-关系-实体
    relation.append([d["name"], d["position"], "play_the_role_of"]) #实体-关系 rolefor-   实体
    relation.append([d["name"], d["country"], "come_from"]) # 实体- 关系- 实体

df = pd.DataFrame(relation)
df.to_csv("newdata/relations.csv", index=False)

with open("newdata/players.json", "w") as file:
#json.dump(total, file) 这行代码的作用是将 Python 对象 total 中的数据以 JSON 格式写入到文件对象 file 中
    json.dump(total, file) 
with open("newdata/country.json", "w") as file:
    json.dump(countrys, file) 
with open("newdata/clubs.json", "w") as file:
    json.dump(club, file) 
with open("newdata/positions.json", "w") as file:
    json.dump(position, file) 

print(len(total))
print(len(countrys))
print(len(club))
print(len(position))

# 744 29 331 14

# data = pd.read_csv("../newdata/relations.csv")
# print(data.values.tolist())