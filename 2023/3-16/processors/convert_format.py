import json
from tqdm import tqdm, trange
import os
from os.path import join
from loguru import logger #log日志
import os
from  utils.utils import write_lines, load_lines  #别用于将一组文本写入到文件中和从文件中读取一组文本。

def bmes_to_json(bmes_file, json_file):
    """
    将bmes格式的文件，转换为json文件，json文件包含text和label,并且转换为BIOS的标注格式
    Args:
        bmes_file:
        json_file:
    :return:
    """

#当读取到的行是非空行时，会执行 else 中的代码，将该行的词语和标签加入 words 和 labels 列表中，
    # 最终将该行的样本以 JSON 格式保存在 texts 列表中。然而在读取最后一行时，由于该行没有回车符 \n，因此 line 仍然包含文本内容，
# 但无法通过 split() 方法将其拆分成 word 和 label 两部分，从而导致了 ValueError: not enough values to unpack (expected 2, got 1) 错误的发生。
    texts = []
    with open(bmes_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        words = []
        labels = []
        num_lines = len(lines)  # 记录文件总行数
        for idx in trange(len(lines)):
            line = lines[idx].strip()
            # if idx==num_lines-1: # 最后一行直接跳过
            #     continue
            if not line:
                assert len(words) == len(labels), (len(words), len(labels)) # 程序结束
                sample = {}
                sample['text'] = words
                sample['label'] = labels
                texts.append(json.dumps(sample, ensure_ascii=False))

                words = []
                labels = []
            else:
                word, label = line.split()
                label = label.replace('M-', 'I-').replace('E-', 'I-')
                words.append(word)
                labels.append(label)

    with open(json_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write("{}\n".format(text))


def bmes_to_json2(bmes_file, json_file):
    result = []
    lines = load_lines(bmes_file)
    for line in lines:
        line = line.strip()
        text, labels = line.split('\t')
        text = text.split()
        labels = [label.replace('M-', 'I-').replace('E-', 'I-') for label in labels.split()]
        assert len(text) == len(labels)
        sample = {'text': text, 'label': labels}
        result.append(json.dumps(sample, ensure_ascii=False))
    write_lines(result, json_file)


def get_label_tokens(input_file, output_file):
    """
    从数据集中获取所有label
    :return:
    """
    labels = set()
    lines = load_lines(input_file)
    for line in lines:
        data = json.loads(line)
        for label in data['label']:
#这行代码是在将 BMES 格式的标签转换为 BIO 格式的标签，具体来说，将 "M-"（中间）和 "E-"（结束）的标签转换为 "I-"（内部）的标签，
# 因为在 BIO 格式中，只有 "B-"（开始）和 "I-"（内部）两种标签，没有 "M-" 和 "E-"。
#BIO标注中也包含"S"标签，"S-"表示的是一个单独的实体，即仅包含一个token。
#BMES -> BIO
            label = label.replace('M-', 'I-').replace('E-', 'I-')
            labels.add(label)
    labels = list(labels)#将输入的标签列表转换成列表类型。
    labels = sorted(labels)#对标签列表进行排序。
    labels.remove('O') # 将'O'标签从列表中删除。
    labels.insert(0, 'O')#将'O'标签插入到列表的开头。
    logger.info('len of label:{}'.format(len(labels)))
    write_lines(labels, output_file) #将结果写入到指定的输出文件中。


# def convert_cner():
#     # bmes生成json
#     bmes_files = ['../datasets1/cner/dev.txt', '../datasets1/cner/test.txt', '../datasets1/cner/train.txt']
#     for bmes_file in bmes_files:
#         dirname = os.path.dirname(bmes_file)
#         file_name = os.path.basename(bmes_file).split('.')[0] + '.json'
#         json_file = join(dirname, file_name)
#         bmes_to_json(bmes_file, json_file)
#
#     # 生成label文件
#     input_file = './datasets1/cner/train.txt'
#     output_file = './datasets1/cner/labels.txt'
#     get_label_tokens(input_file, output_file)


# def convert_weibo():
#     # bmes生成json
#     bmes_files = ['../datasets1/weibo/dev.char.bmes', '../datasets/weibo/test.char.bmes', '../datasets/weibo/train.char.bmes']
#     for bmes_file in bmes_files:
#         dirname = os.path.dirname(bmes_file)
#         file_name = os.path.basename(bmes_file).split('.')[0] + '.json'
#         json_file = join(dirname, file_name)
#         bmes_to_json(bmes_file, json_file)
#
#     # 生成label文件
#     input_file = '../datasets1/weibo/train.char.bmes'
#     output_file = '../datasets1/weibo/labels.txt'
#     get_label_tokens(input_file, output_file)


if __name__ == '__main__':
    #生成json文件
#     data_names = ['msra', 'ontonote4', 'resume', 'weibo']
#     path = '/root/autodl-tmp/project/datasets1/ner_data'
#     for data_name in data_names:
#         logger.info('processing dataset:{}'.format(data_name))
# #这行代码的作用是获取指定路径下data_name文件夹中的所有文件名，返回一个包含所有文件名的列表。
#         for data_type in ['train','test','dev']:
#             # data_type='test'# train ,test,l
#             if data_name =='msra':
#                 file=join(path,data_name,data_type+'.bmes')
#                 out_path = join(path, data_name, data_type+'.json')
#                 bmes_to_json(file, out_path)
#             elif data_name=='weibo':
#                 file = join(path, data_name, data_type +'.all'+ '.bmes')
#                 out_path = join(path, data_name, data_type + '.json')
#                 bmes_to_json(file, out_path)
#             else:
#                 file = join(path, data_name, data_type + '.char' + '.bmes')
#                 out_path = join(path, data_name, data_type + '.json')
#                 bmes_to_json2(file, out_path)


   # 生成label文件
    data_names = ['msra', 'ontonote4', 'resume', 'weibo']
    path = '/root/autodl-tmp/project/datasets1/ner_data'
    for data_name in data_names:
        logger.info('processing dataset:{}'.format(data_name))
        input_file = join(path, data_name, 'train.json')
        output_file = join(path, data_name, 'labels.txt')
        get_label_tokens(input_file, output_file)
