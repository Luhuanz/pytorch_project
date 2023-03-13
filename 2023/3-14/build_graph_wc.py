import os
import json
import pandas as pd
from py2neo import Graph, Node, Relationship


class CharacterGraph:
    def __init__(self, path):
        self.path = path
        self.g = Graph("bolt://44.204.48.28:7687", auth=("neo4j", "defect-insanity-environments"))
    
    def create_nodes(self, label): #label from clubs/country/positions
        print("import {} nodes".format(label))
        with open(os.path.join(self.path, "{}.json".format(label))) as f:
            data = json.load(f)
        for i, d in enumerate(data):
            node = Node(label, name=d)
            self.g.create(node)
            print("{}-th node of {}".format(i, label))
    
    def create_player_nodes(self):
        print("import player nodes")
        with open(os.path.join(self.path, "players.json")) as f:
            data = json.load(f)
        for i, d in enumerate(data):
            node = Node("players", name=d["name"], number=d["number"], age=d["age"], apperance=d["apperance"], goal=d["goal"])
            self.g.create(node)
            print("{}-th node of players".format(i))

    def create_all_nodes(self):
        self.create_player_nodes()
        self.create_nodes("clubs")
        self.create_nodes("country")
        self.create_nodes("positions")      

    def create_edges(self):
        data = pd.read_csv(os.path.join(self.path, "relations.csv"))
        data = data.values.tolist()

        relation_dict = {"work_for":"clubs", "play_the_role_of":"positions", "come_from":"country"}
        for i, d in enumerate(data):
            a = self.g.nodes.match("players", name=d[0]).first()
            b = self.g.nodes.match(relation_dict[d[2]], name=d[1]).first()
            r = Relationship(a, d[2], b)
            self.g.create(r)
            print("{}-th edges".format(i))
    
    def create_txt_realtions(self):
        with open(os.path.join(self.path, "players.json")) as f:
            players = json.load(f)
        with open(os.path.join(self.path, "clubs.json")) as f:
            clubs = json.load(f)
        with open(os.path.join("nerdata/final_relations.txt")) as f:
            rel = f.read().split("\n")
        
        for i, r in enumerate(rel):
            r = r.split(" ")
            name, org = r[0], r[1]
            if name not in players:
                node = Node("players", name=name)
                self.g.create(node)
                players.append(name)
                print("Create new players node {}".format(name))
            if org not in clubs:
                node = Node("clubs", name=org)
                self.g.create(node)
                clubs.append(org)
                print("Create new club node {}".format(org))
            a = self.g.nodes.match("players", name=name).first()
            b = self.g.nodes.match("clubs", name=org).first()
            r = Relationship(a, "used_to_work_for", b)
            self.g.create(r)
            print("{}-th new edges".format(i))

if __name__ == '__main__':
    handler = CharacterGraph("newdata")
    print("step1:导入图谱节点中")
    handler.create_all_nodes()
    print("step2:导入图谱边中")
    handler.create_edges()

    # import relations from NER!!!!!
    print("step3:导入新的关系")
    handler.create_txt_realtions()
