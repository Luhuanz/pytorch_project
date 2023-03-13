# pipelines.py 用来将爬取的内容存放到 MySQL 数据库中。
# 类内有初始化 init()、处理爬取内容并保存 process_item ()、
# 关闭数据库 close_spider () 三个方法。
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pymysql.cursors import Cursor
#import sys
#from importlib import reload

#reload(sys)
#sys.setdefaultencoding('utf-8')

import pymysql
from pymysql import connections
from baidu_baike import settings

class BaiduBaikePipeline(object):
    def __init__(self):
        # 初始化并连接到mysql数据库
        self.conn = pymysql.connect(
            host=settings.HOST_IP,
            port=settings.PORT,
            user=settings.USER,
            passwd=settings.PASSWD,
            db=settings.DB_NAME,
            charset='utf8mb4', #utf8mb4 是一种 Unicode 编码格式，它支持存储最广泛使用的字符集，包括 Emoji 表情等多种语言和符号。如果您的应用程序需要处理多语言字符，则建议使用 utf8mb4 字符集编码。
            use_unicode=True
            )
        self.cursor = self.conn.cursor(Cursor)
#用于处理爬虫爬取到的数据。其中，item 参数是爬虫爬取到的数据，spider 参数是爬虫对象。
    def process_item(self, item, spider):
        # process info for actor
#在处理数据时，代码将 actor_chName、actor_foreName、movie_chName、movie_foreName 这些字段转换成了 utf-8 编码，并存储到变量中。
        actor_chName = str(item['actor_chName']).encode('utf-8')
        actor_foreName = str(item['actor_foreName']).encode('utf-8')
        movie_chName = str(item['movie_chName']).encode('utf-8')
        movie_foreName = str(item['movie_foreName']).encode('utf-8')
#这段代码中，根据条件判断，如果 item 中的 actor_chName 或 actor_foreName 字段不为 None，并且 movie_chName 字段为 None，则执行下面的代码。
        if (item['actor_chName'] != None or item['actor_foreName'] != None) and item['movie_chName'] == None:
            actor_bio = str(item['actor_bio']).encode('utf-8')
            actor_nationality = str(item['actor_nationality']).encode('utf-8')
            actor_constellation = str(item['actor_constellation']).encode('utf-8')
            actor_birthPlace = str(item['actor_birthPlace']).encode('utf-8')
            actor_birthDay = str(item['actor_birthDay']).encode('utf-8')
            actor_repWorks = str(item['actor_repWorks']).encode('utf-8')
            actor_achiem = str(item['actor_achiem']).encode('utf-8')
            actor_brokerage = str(item['actor_brokerage']).encode('utf-8')
# 用于从数据库中查询演员姓名（actor_chName 字段）并将结果存储到 actorList 变量中。
            self.cursor.execute("SELECT actor_chName FROM actor;")
            actorList = self.cursor.fetchall()
    #将爬取到的演员信息存储到数据库中
            #检查演员是否已经存在于数据库中。如果不存在，就执行下面的操作；否则，跳过该操作。
            if (actor_chName,) not in actorList :
                # get the nums of actor_id in table actor
                #查询当前 actor 表中 actor_id 字段的最大值。如果表中还没有任何记录，则结果为 None。
                self.cursor.execute("SELECT MAX(actor_id) FROM actor")
                result = self.cursor.fetchall()[0]
                if None in result:
                    actor_id = 1
                else:
                    actor_id = result[0] + 1
                sql = """
                INSERT INTO actor(actor_id, actor_bio, actor_chName, actor_foreName, actor_nationality, actor_constellation, actor_birthPlace, actor_birthDay, actor_repWorks, actor_achiem, actor_brokerage ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            # 写入
                self.cursor.execute(sql, (actor_id, actor_bio, actor_chName, actor_foreName, actor_nationality, actor_constellation, actor_birthPlace, actor_birthDay, actor_repWorks, actor_achiem, actor_brokerage ))
                self.conn.commit()
            else:
                print("#" * 20, "Got a duplict actor!!", actor_chName)
#这段代码检查了是否有电影信息（中文电影名或外文电影名），但没有演员信息。如果电影信息存在且演员信息不存在，
        # 将从item字典中获取电影信息，编码为UTF-8，并将其存储到对应的变量中。
        elif (item['movie_chName'] != None or item['movie_foreName'] != None) and item['actor_chName'] == None:
            movie_bio = str(item['movie_bio']).encode('utf-8') #电影的简介
            movie_prodTime = str(item['movie_prodTime']).encode('utf-8') #电影的制作时间
            movie_prodCompany = str(item['movie_prodCompany']).encode('utf-8') #电影的制片公司
            movie_director = str(item['movie_director']).encode('utf-8')
            movie_screenwriter = str(item['movie_screenwriter']).encode('utf-8')
            movie_genre = str(item['movie_genre']).encode('utf-8')
            movie_star = str(item['movie_star']).encode('utf-8')
            movie_length = str(item['movie_length']).encode('utf-8')
            movie_rekeaseTime = str(item['movie_rekeaseTime']).encode('utf-8')
            movie_language = str(item['movie_language']).encode('utf-8')
            movie_achiem = str(item['movie_achiem']).encode('utf-8') #电影的成就

            self.cursor.execute("SELECT movie_chName FROM movie;")
            movieList = self.cursor.fetchall()
            if (movie_chName,) not in movieList :
                self.cursor.execute("SELECT MAX(movie_id) FROM movie")
                result = self.cursor.fetchall()[0]
                if None in result:
                    movie_id = 1
                else:
                    movie_id = result[0] + 1
                sql = """
                INSERT INTO movie(  movie_id, movie_bio, movie_chName, movie_foreName, movie_prodTime, movie_prodCompany, movie_director, movie_screenwriter, movie_genre, movie_star, movie_length, movie_rekeaseTime, movie_language, movie_achiem ) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                self.cursor.execute(sql, ( movie_id, movie_bio, movie_chName, movie_foreName, movie_prodTime, movie_prodCompany, movie_director, movie_screenwriter, movie_genre, movie_star, movie_length, movie_rekeaseTime, movie_language, movie_achiem ))
                self.conn.commit()
            else:
                print("Got a duplict movie!!", movie_chName)
        else:
            print("Skip this page because wrong category!! ")
        return item
    def close_spider(self, spider):
        self.conn.close()
