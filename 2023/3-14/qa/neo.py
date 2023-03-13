import csv
import py2neo
from py2neo import Graph,Node,Relationship,NodeMatcher

class NeoDB(object):
    def __init__(self):
        self.graph = Graph("bolt://18.212.17.194:7687", auth=("neo4j", "operand-fruits-hood"))
        self.similar_words = {
            "进球数": "goal", "年龄": "age"
        }

        return