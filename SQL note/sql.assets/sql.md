# SQL

## Select `查`

### DISTINCT 

**DISTINCT 关键词用于返回唯一不同的值。**

```SQL
SELECT DISTINCT country FROM Websites; 
```

 返回websites表中country值（唯一值 CN CN 只是CN）

###  WHERE 

**WHERE 子句用于提取那些满足指定条件的记录**。

```sql
SELECT * FROM Websites WHERE country='CN'; 
```

**空值判断： is null**

```sql
Select * from emp where comm is null;
```

**between and (在 之间的值)** ：**查询 emp 表中 SAL 列中大于 1500 的小于 3000 的值。**

```sql
Select * from emp where sal between 1500 and 3000;
```

**In**  :**查询 EMP 表 SAL 列中等于 5000，3000，1500 的值。**

```sql
Select * from emp where sal in (5000,3000,1500);
```

NOT IN：

```sql
SELECT 字段(s)
FROM 表名
WHERE 条件字段 NOT IN (v1,v2,...);
<=>
SELECT 字段(s)
FROM 表名
WHERE 条件字段 <> v1 AND 条件字段 <> v2 ...;
```

**like**: **Like模糊查询** **查询 EMP 表中 Ename 列M开头的全部 ，M 为要查询内容中的模糊信息。**

```sql
Select * from emp where ename like 'M%';
```

**选取 name 列 以字母 "k" 结尾的所有客户：**

```sql
SELECT * FROM Websites
WHERE name LIKE '%k';
```

**选取 name 包含模式 "oo" 的所有客户**：

```sql
SELECT * FROM Websites
WHERE name LIKE '%oo%';
```

-  **%** 表示多个字值，**_** 下划线表示一个字符；
-  **M%** : 为能配符，正则表达式，表示的意思为模糊查询信息为 M 开头的。
-  **%M%** : 表示查询包含M的所有内容。
-  **%M_** : 表示查询以M在倒数第二位的所有内容。
- '**_a_**'   :三位且中间字母是a的
- '_**a** '   :两位且结尾字母是a的
- '**a**_'   :两位且开头字母是a的

| 运算符 | =    | <>     | >    | <    | >=       | <=       | BETWEEN      | LIKE         | IN                         |
| ------ | ---- | ------ | ---- | ---- | -------- | -------- | ------------ | ------------ | -------------------------- |
| 描述   | 等于 | 不等于 | 大于 | 小于 | 大于等于 | 小于等于 | 在某个范围内 | 搜索某种模式 | 指定针对某个列的多个可能值 |

### between&not between

**选取 alexa 介于 1 和 20 之间的所有网站：**

```sql
SELECT * FROM Websites
WHERE alexa BETWEEN 1 AND 20;
```

**选取 alexa 不介于 1 和 20 之间的所有网站：**

```sql
SELECT * FROM Websites
WHERE alexa NOT BETWEEN 1 AND 20;
```

**选取 alexa 介于 1 和 20 之间但 country 不为 USA 和 IND 的所有网站：**

```sql
SELECT * FROM Websites
WHERE (alexa BETWEEN 1 AND 20)
AND country NOT IN ('USA', 'IND');
```



### AND & OR

如果第一个条件和第二个条件都成立，则 AND 运算符显示一条记录。

如果第一个条件和第二个条件中只要有一个成立，则 OR 运算符显示一条记录。

**从 "Websites" 表中选取国家为 "CN" 且alexia 排名大于 "50" 的所有网站：**

```SQL
         SELECT * FROM Websites WHERE country='CN' AND alexa > 50; 
```

**从 "Websites" 表中选取国家为 "USA" 或者 "CN" 的所有客户：**

```sql
SELECT * FROM Websites WHERE country='USA' OR country='CN';
```

**从 "Websites" 表中选取 alexia 排名大于 "15" 且国家为 "CN" 或 "USA" 的所有网站：**

```sql
SELECT * FROM Websites WHERE alexa > 15 AND (country='CN' OR country='USA');
```

###  order by 排序

 用于对结果集按照一个列或者多个列进行排序。

 默认按照升序；降序使用 DESC；

**从 "Websites" 表中选取全部列，并按照 "alexa" 列排序：** 

```sql
SELECT * FROM Websites ORDER BY alexa;
```

**"Websites" 表中选取全部列，并按照 "alexa" 列降序排序：**

```sql
SELECT * FROM Websites ORDER BY alexa DESC;
```

 **"Websites" 表中选取所有网站，并按照 "country" 和 "alexa" 列排序：**

ORDER BY 多列的时候，先按照第一个column name排序，在按照第二个column name排序； 

-  1）、先将country值这一列排序，同为CN的排前面，同属USA的排后面；
-  2）、然后在同属CN的这些多行数据中，再根据alexa值的大小排列。
-  3）、ORDER BY 排列时，不写明ASC ,DESC的时候，默认是ASC。

```sql
SELECT * FROM Websites ORDER BY country,alexa;
```

**高级**

### LIMIT

用于规定要返回的记录的数目。(对于拥有数千条记录的大型表来说，是非常有用的。)

**从 "Websites" 表中选取头两条记录：**

```sql
SELECT * FROM Websites LIMIT 2;
```

```sql
select top 5 * from table 

--后5行
select top 5 * from table order by id desc  --desc 表示降序排列 asc 表示升序
```

### **REGEXP** 或 **NOT REGEXP** 运算符 (或 RLIKE 和 NOT RLIKE)  正则表达式

**选取 name 以 "G"、"F" 或 "s" 开头的所有网站：**

```sql
SELECT * FROM Websites
WHERE name REGEXP '^[GFs]';
```

**选取 name 不以 A 到 H 字母开头的网站：** **^** 以开头

```sql
SELECT * FROM Websites
WHERE name REGEXP '^[^A-H]';
```

### 别名

一个是 name 列的别名，一个是 country 列的别名

```sql
SELECT name AS n, country AS c
FROM Websites;
```

使用 "Websites" 和 "access_log" 表，并分别为它们指定表别名 "w" 和 "a"

```sql
SELECT w.name, w.url, a.count, a.date
FROM Websites AS w, access_log AS a
WHERE a.site_id=w.id and w.name="菜鸟教程";
```

### join（非常重要）

![img](/sql-join.png)

SQL JOIN 子句用于把来自两个或多个表的行结合起来，基于这些表之间的共同字段。

​	Websites" 表中的 "**id**" 列指向 "access_log" 表中的字段 "**site_id**"。上面这两个表是通过 "site_id" 列联系起来的。

```SQL
SELECT Websites.id, Websites.name, access_log.count, access_log.date
FROM Websites
INNER JOIN access_log
ON Websites.id=access_log.site_id;
```

### Union

UNION 操作符用于合并两个或多个 SELECT 语句的结果集。 UNION ALL  拿出重复(不去重)

**从 "Websites" 和 "apps" 表中选取所有不同的country（只有不同的值）：**

```SQL
SELECT country FROM Websites
UNION
SELECT country FROM apps
ORDER BY country;
```

**从 "Websites" 和 "apps" 表中选取所有的country（也有重复的值）：**

```sql
SELECT country FROM Websites
UNION ALL
SELECT country FROM apps
ORDER BY country;
```

## 常用函数 select用

### AVG()

 **"access_log" 表的 "count" 列获取平均值：**

```sql
SELECT AVG(count) AS CountAverage FROM access_log;
```

**选择访问量高于平均访问量的 "site_id" 和 "count"：**

```sql
SELECT site_id, count FROM access_log
WHERE count > (SELECT AVG(count) FROM access_log);
```

### Count()

COUNT() 函数返回匹配指定条件的行数。

**计算 "access_log" 表中 "site_id"=3 的总访问量：**

```SQL
Select Count(count) from access_log where site_id=3;
```

### group by

**access_log 各个 site_id 的访问量：**

```sql
SELECT site_id, SUM(access_log.count) AS nums
FROM access_log GROUP BY site_id;
```

### Having

HAVING 子句原因是，WHERE 关键字无法与聚合函数一起使用。

**想要查找总访问量大于 200 的网站。**

```sql
SELECT Websites.name, Websites.url, SUM(access_log.count) AS nums FROM (access_log
INNER JOIN Websites
ON access_log.site_id=Websites.id)
GROUP BY Websites.name
HAVING SUM(access_log.count) > 200;
```

**查找总访问量大于 200 的网站，并且 alexa 排名小于 200。**

```sql
SELECT Websites.name, SUM(access_log.count) AS nums FROM Websites
INNER JOIN access_log
ON Websites.id=access_log.site_id
WHERE Websites.alexa < 200 
GROUP BY Websites.name
HAVING SUM(access_log.count) > 200;
```

where 和having之后都是筛选条件，但是有区别的：

1.where在group by前， having在group by 之后

2.聚合函数（avg、sum、max、min、count），不能作为条件放在where之后，但可以放在having之后



## insert into 增

### insert into

INSERT INTO 语句用于向表中插入新记录。

**向 "Websites" 表中插入一个新行:(不指定列名)**

```sql
INSERT INTO Websites (name, url, alexa, country) VALUES ('百度','https://www.baidu.com/','4','CN');
```

指定的列插入数据

**插入一个新行，但是只在 "name"、"url" 和 "country" 列插入数据**

```sql
INSERT INTO Websites (name, url, country)
VALUES ('stackoverflow', 'http://stackoverflow.com/', 'IND');
```



##  update 改

UPDATE 语句用于更新表中已存在的记录。

**把 "菜鸟教程" 的 alexa 排名更新为 5000，country 改为 USA。**

```sql
UPDATE Websites  SET alexa='5000', country='USA'  WHERE name='菜鸟教程';
```



## delete 删

**Websites" 表中删除网站名为 "Facebook" 且国家为 USA 的网站。**

```sql
DELETE FROM Websites WHERE name='Facebook' AND country='USA';
```