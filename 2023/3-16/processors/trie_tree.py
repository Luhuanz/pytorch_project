import collections
#这段 Python 代码实现了一个 Trie（字典树）数据结构，并定义了该数据结构的一些基本操作，
# 为了实现字典对匹配
class TrieNode:
    def __init__(self):
        # children是一个字典，键是字符，值是TrieNode类对象，用于保存当前节点的子节点
        self.children = collections.defaultdict(TrieNode)
    ## is_word是一个布尔型变量，表示从根节点到当前节点的路径上是否构成一个完整的单词
        self.is_word = False

class Trie:
    """
    In fact, this Trie is a letter three.
    root is a fake node, its function is only the begin of a word, same as <bow>
    the the first layer is all the word's possible first letter, for example, '中国'
        its first letter is '中'
    the second the layer is all the word's possible second letter.
    and so on
    """
## 初始化函数，use_single表示是否将单字词也作为一个完整的单词进行保存，如果是True，就将单字词也视为单词
    def __init__(self, use_single=True):
# root是一个TrieNode类对象，表示字典树的根节点，由于根节点不表示任何字符，因此其children为空
        self.root = TrieNode()
# max_depth是一个整数，表示字典树的最大深度
        self.max_depth = 0
# 如果use_single为True，就将min_len设置为0，否则将min_len设置为1
        if use_single:
            self.min_len = 0
        else:
            self.min_len = 1

    # insert方法用于向字典树中插入一个单词
    def insert(self, word):
        # current是一个TrieNode类对象，表示当前节点，初始时为根节点
        current = self.root
        # deep是一个整数，表示从根节点到当前节点的路径长度，初始值为0
        deep = 0
        # 遍历word中的每一个字符
        for letter in word:
            # 将current更新为当前字符的节点
            current = current.children[letter]
            # 将deep加1
            deep += 1
            # 将current的is_word属性设置为True，表示从根节点到当前节点的路径上构成了一个完整的单词
        current.is_word = True
        # 如果当前路径长度大于max_depth，就将max_depth更新为当前路径长度
        if deep > self.max_depth:
            self.max_depth = deep

    # search方法用于在字典树中查找一个单词
    def search(self, word):
        # current是一个TrieNode类对象，表示当前节点，初始时为根节点
        current = self.root
        # 遍历word中的每一个字符
        for letter in word:

            # 将current更新为当前字符的节点
            current = current.children.get(letter)
            # 如果current为None，表示当前单词在字典树中不存在，直接返回False
            if current is None:
                return False
        return current.is_word

    def enumerateMatch(self, str, space=""):
        """
        Args:
            str: 需要匹配的词
        Return:
            返回匹配的词, 如果存在多字词，则会筛去单字词
        """
        matched = []
        while len(str) > self.min_len:
            if self.search(str):
                matched.insert(0, space.join(str[:])) # 短的词总是在最前面
            del str[-1]

        if len(matched) > 1 and len(matched[0]) == 1: # filter single character word
            matched = matched[1:]

        return matched

