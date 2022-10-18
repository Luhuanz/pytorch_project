import random
from decimal import *
import numpy as np
import collections
from tqdm import tqdm


class VoseAlias(object):
    """
    Adding a few modifs to https://github.com/asmith26/Vose-Alias-Method
    """

    def __init__(self, dist):
        """
        (VoseAlias, dict) -> NoneType
        """
        self.dist = dist
        self.alias_initialisation()

    def alias_initialisation(self):
        """
        Construct probability and alias tables for the distribution.
        """
        # Initialise variables
        n = len(self.dist)
        self.table_prob = {}   # probability table概率表
        self.table_alias = {}  # alias table替身表
        scaled_prob = {}       # scaled probabilities乘以n的概率表
        small = []             # stack for probabilities smaller that 1存储概率值小于1的
        large = []             # stack for probabilities greater than or equal to 1存储概率值大于1的

        # Construct and sort the scaled probabilities into their appropriate stacks
        print("1/2. Building and sorting scaled probabilities for alias table...")
        for o, p in tqdm(self.dist.items()):
            scaled_prob[o] = Decimal(p) * n

            if scaled_prob[o] < 1:
                small.append(o)
            else:
                large.append(o)

        print("2/2. Building alias table...")
        # Construct the probability and alias tables
        # 使用贪心算法，将概率值小于1的列表不断填成1
        while small and large:
            s = small.pop()
            l = large.pop()

            self.table_prob[s] = scaled_prob[s]
            self.table_alias[s] = l

            scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - Decimal(1)

            if scaled_prob[l] < 1:
                small.append(l)
            else:
                large.append(l)

        # The remaining outcomes (of one stack) must have probability 1
        # 当两方不全有元素时，仅有一方有元素的也全为1
        # 就是最后一个large列表中的那个元素应该为1
        while large:
            self.table_prob[large.pop()] = Decimal(1)

        while small:
            self.table_prob[small.pop()] = Decimal(1)
        self.listprobs = list(self.table_prob)

    def alias_generation(self):
        """
        Yields a random outcome from the distribution.
        """
        # Determine which column of table_prob to inspect
        col = random.choice(self.listprobs)
        # Determine which outcome to pick in that column
        # 取自己
        if self.table_prob[col] >= random.uniform(0, 1):
            return col
        # 取替身
        else:
            return self.table_alias[col]

    def sample_n(self, size):
        """
        Yields a sample of size n from the distribution, and print the results to stdout.
        """
        # 调用alias generation一共n次，采样n个nodes
        for i in range(size):
            yield self.alias_generation()

#读图函数
#初始化词典
def makeDist(graphpath, power=0.75):

    edgedistdict = collections.defaultdict(int)
    nodedistdict = collections.defaultdict(int)

    weightsdict = collections.defaultdict(int)
    nodedegrees = collections.defaultdict(int)

    # 用来做归一化的两个sum变量
    weightsum = 0
    negprobsum = 0

    nlines = 0 #统计图一共有多少条边

    with open(graphpath, "r") as graphfile:
        for l in graphfile:
            nlines += 1

    print("Reading edgelist file...")
    maxindex = 0
    with open(graphpath, "r") as graphfile:
        # #用qdm展示for循环进度百分比
        for l in tqdm(graphfile, total=nlines):
            #将\n换行符去掉，并按空格分词，存储格式为：点i，点j，weight
            line = [int(i) for i in l.replace("\n", "").split(" ")]
            node1, node2, weight = line[0], line[1], line[2]

            # 后面会做归一化，存的是归一化的边-权重和点-出度
            edgedistdict[tuple([node1, node2])] = weight
            nodedistdict[node1] += weight

            # 不再做处理，存的是边-权重，点-出度
            weightsdict[tuple([node1, node2])] = weight
            nodedegrees[node1] += weight

            # weightsum存的是全图所有边的边权和，论文公式（2）中用到的1st相似度真实值
            weightsum += weight
            negprobsum += np.power(weight, power)

            # maxindex记录图中最大顶点index
            if node1 > maxindex:
                maxindex = node1
            elif node2 > maxindex:
                maxindex = node2

    for node, outdegree in nodedistdict.items():
        nodedistdict[node] = np.power(outdegree, power) / negprobsum

    for edge, weight in edgedistdict.items():
        edgedistdict[edge] = weight / weightsum
    # edgedistdict边且归一化
    # nodedistdict点且归一化
    # weightsdict边的权重
    # nodedegrees点的出度
    # maxindex最大节点index
    return edgedistdict, nodedistdict, weightsdict, nodedegrees, maxindex


def negSampleBatch(sourcenode, targetnode, negsamplesize, weights,
                   nodedegrees, nodesaliassampler, t=10e-3):
    """
    For generating negative samples.
    """
    negsamples = 0
    while negsamples < negsamplesize:#negsamplesize我们设置的负样本是5个点，取够5个才停止
        # nodesaliassampler是实现alias building的VoseAlias类，这里采样点
        samplednode = nodesaliassampler.sample_n(1)
        # 如果采样出source或target均跳过
        if (samplednode == sourcenode) or (samplednode == targetnode):
            continue
        else:#辅出负样本点，一共negsamplesize个点
            negsamples += 1
            yield samplednode


def makeData(samplededges, negsamplesize, weights, nodedegrees, nodesaliassampler):
    for e in samplededges:# 遍历samplededges
        sourcenode, targetnode = e[0], e[1]#起点和终点
        negnodes = []
        # 采样出negsamplesize（5）个负样本点
        for negsample in negSampleBatch(sourcenode, targetnode, negsamplesize,
                                        weights, nodedegrees, nodesaliassampler):
            for node in negsample:#将所有的负样本点加入到negnodes列表中
                negnodes.append(node)
        # 格式是（node i，node j，negative nodes..…）总共7个点，前面两个正样本边的点i和j，后面5个是负样本
        yield [e[0], e[1]] + negnodes
