# 已经失效，维基百科更新
from lxml import etree
import urllib.request
import urllib.parse
import pandas as pd
import json
import numpy as np

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36' }

def save_json(save_path,data):   #save_path 表示 JSON 文件保存路径的字符串  data 要保存在保存在 JSON 文件中的数据。
    assert save_path.split('.')[-1] == 'json'  #如果文件扩展名不是 "json"，那么就会触发 assert 关键字后面的错误，提示用户输入一个正确的 JSON 文件路径。
    with open(save_path,'w',encoding='utf-8') as file:
        json.dump(data,file) #，用于将 Python 对象序列化为 JSON 格式并将其写入文件。

def getTeams():
    url = 'https://m.tw.cljtscd.com/baike-2022%E5%B9%B4%E4%B8%96%E7%95%8C%E6%9D%AF'
    req = urllib.request.Request(url=url, headers=headers, method='GET')
    response = urllib.request.urlopen(req)
    text = response.read().decode('utf-8')
    html = etree.HTML(text)
    country_list = html.xpath('//table[@class="multicol"]//li/a/text()') #澳大利亚 伊朗 日本 国家列表
    country_href = html.xpath('//table[@class="multicol"]//li//@href') # 不同国家队伍的具体网站

    return country_list, country_href
#接收一个参数 url，用于从指定的网址获取个人信息。
def getPersonalInfo(url):
    try:
        req = urllib.request.Request(url=url, headers=headers, method='GET')
        response = urllib.request.urlopen(req)
        text = response.read().decode('utf-8')
        html = etree.HTML(text)
        table = html.xpath('//table[@class="infobox vcard"]')
        if len(table)==0:
            return []
        table = etree.tostring(table[0], encoding='utf-8').decode()
        df = pd.read_html(table, encoding='utf-8', header=0)[0]
  
        personal_info_list = [
            '全名', 
            '暱稱',
            '出生地點',
            '身高',
            '位置',
            '現在所屬',
            '球衣號碼',
            '榮譽'
            ]

        rows,cols = df.shape
        personal_info = []
        for i in personal_info_list:
            for r in range(rows):
                if df.loc[r][0]==i:
                    personal_info.append({i:df.loc[r][2]})

        bday = html.xpath('//table[@class="infobox vcard"]//span[@class="bday"]/text()')[0]
        personal_info.append({'出生日期':bday})
    
        for r in range(rows):
            if isinstance(df.loc[r][0],str) and len(df.loc[r][0])>2 and df.loc[r][0][:2]=='榮譽':
                honor = df.loc[r][0]
                personal_info.append({'荣誉':honor[honor.find('男子足球')+5:]})

        return personal_info
    
    except ValueError :
        return []
    except TimeoutError:
        return []
    except urllib.error.URLError:
        return [] 
# 获取俱乐部
def get_clubs(url):
    try:
        req = urllib.request.Request(url=url, headers=headers, method='GET')
        response = urllib.request.urlopen(req)
        text = response.read().decode('utf-8')

        html = etree.HTML(text)
        club = html.xpath('//*[@id="mf-section-1"]//p//text()')
        if len(club)==0:
            return ''
        club_pd = pd.DataFrame(club)
        club_pd= club_pd[~ club_pd[0].str.contains(r'(\[)+',regex=True)]
        clubs = np.array(club_pd[0]).tolist()
        whole = ''
        for i in range(len(clubs)):
            whole+=clubs[i]
        return whole
    
    except ValueError :
        return ""
    except TimeoutError:
        return ""
    except urllib.error.URLError:
        return "" 
#从给定的 url 网页中提取足球运动员信息，并将其保存为 JSON 格式的文件。
def query(url, country):
    req = urllib.request.Request(url=url, headers=headers, method='GET')
    response = urllib.request.urlopen(req)
    text = response.read().decode('utf-8')

    html = etree.HTML(text)
    tables = html.xpath('//table')
    for i in range(len(tables)):
        table = etree.tostring(tables[i], encoding='utf-8').decode() #用于将一个 Element 对象或一个 ElementTree 序列化为一个 XML 字符串表示
        df = pd.read_html(table, encoding='utf-8', header=0)[0] #是 Pandas 库中的一个函数，用于从一个 HTML 字符串或文件中读取表格数据并将其转换为一个 Pandas 数据帧对象。它返回一个列表
        if (df.columns[0] == '号码' or df.columns[0] == '號碼') and df.shape[0]>1:
            players = list(df.T.to_dict().values())
            for r in range(df.shape[0]):
                name = df.loc[r][2]
                if not isinstance(name,str):
                    continue
                name = name[:name.find('（')]
                player = html.xpath('//a[contains(string(),"'+name+'")]/@href')
                if len(player)>0:
                    players[r]['详细信息'] = getPersonalInfo(player[0])
            print(players)
            save_json(country+'.json',players)
            break


def query_clubs(url, country):
    req = urllib.request.Request(url=url, headers=headers, method='GET')
    response = urllib.request.urlopen(req)
    text = response.read().decode('utf-8')

    html = etree.HTML(text)
    tables = html.xpath('//table')
    for i in range(len(tables)):
        table = etree.tostring(tables[i], encoding='utf-8').decode()
        df = pd.read_html(table, encoding='utf-8', header=0)[0]
        if (df.columns[0] == '号码' or df.columns[0] == '號碼') and df.shape[0]>1:
            players = list(df.T.to_dict().values())
            clubs_ = ''
            for r in range(df.shape[0]):
                name = df.loc[r][2]
                if not isinstance(name,str):
                    continue
                name = name[:name.find('（')]
                player = html.xpath('//a[contains(string(),"'+name+'")]/@href')
                if len(player)>0:
                    clubs_ += get_clubs(player[0])+"\n"
                print(name)
            with open('./clubs/'+country+'.txt', 'w', encoding='utf-8') as f:
                f.write(clubs_)
                f.close()
            break

if __name__ == '__main__':
    country_list, country_href = getTeams()

    for country in range(len(country_list)):
        print(country_list[country])
        query(country_href[country], country_list[country])
        query_clubs(country_href[country], country_list[country])
        
