from neo import NeoDB
import re
import jieba
import jieba.posseg as pseg
import logging
import paddle

logging.basicConfig(level=logging.ERROR)


class KGQA(object):
    def __init__(self):
        super().__init__()
        self.neo = NeoDB()
        paddle.enable_static()
        jieba.enable_paddle()

    def cut_words(self, sentence):
        words_flags = pseg.cut(sentence, use_paddle=True)  # paddle模式

        person = ''
        words = []
        for word, flag in words_flags:
            if flag == 'PER':
                person = word
            if flag == 'n':
                words.append(self.neo.similar_words[word])
        logging.debug(str(words))
        return person, words

    def answer(self, sentence):
        
        try:  
            sentence = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", sentence)
            sentence = re.sub('\W+', '', sentence).replace("_", '')
            person, words = self.cut_words(sentence)

            words_ = [0 for i in range(len(words))]
            for i in range(len(words)):
                words_[i] = '-[r'+str(i) + ':'+words[len(words)-i-1]+']'
                if i != len(words)-1:
                    words_[i] += '->(n'+str(i)+':Person)'
            quary = "match (p:players{name: " + person + '"}) ' + \
                "return p."+ words[0]
        except Exception as e:
            return '问题有误'
        try:
            import ipdb
            ipdb.set_trace()
            data = self.neo.graph.run(quary)
            data = list(data)[0]
            logging.debug(str(data))
            result = person +'的'+ words[0] +'是'+data["p."+words[0]]
            
        except Exception as e:
            return '没有答案'

    def test(self, sentence):
        answer = self.answer(sentence)
        print(answer)

