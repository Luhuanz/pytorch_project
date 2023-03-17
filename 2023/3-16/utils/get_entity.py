def get_entity_bios(seq,id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = [] # 用于存储识别出的命名实体
    chunk = [-1, -1, -1] # 遍历输入的标注序列
    for indx, tag in enumerate(seq):# 如果标注不是字符串类型，就将其转化为标注的名称
        if not isinstance(tag, str): # 如果标注不是字符串类型，就将其转化为标注的名称
            tag = id2label[tag]
        if tag.startswith("S-"):# 如果标注以"S-"开头，则当前标注为单一实体
            if chunk[2] != -1: # 如果当前识别的实体不是第一个实体，则将其存入chunks列表中
                chunks.append(chunk)
            chunk = [-1, -1, -1] # 重置chunk列表的值
            chunk[1] = indx# 当前实体的起始位置
            chunk[2] = indx# 当前实体的结束位置
            chunk[0] = tag.split('-')[1] # 当前实体的类型
            chunks.append(chunk)  # 将当前实体存入chunks列表
            chunk = (-1, -1, -1) # 重置chunk列表的值
        if tag.startswith("B-"):# 如果标注以"B-"开头，则当前标注为一个实体的开始
            if chunk[2] != -1:# 如果当前识别的实体不是第一个实体，则将其存入chunks列表中
                chunks.append(chunk)
            chunk = [-1, -1, -1]# 重置chunk列表的值
            chunk[1] = indx# 当前实体的起始位置
            chunk[0] = tag.split('-')[1]# 当前实体的类型
        elif tag.startswith('I-') and chunk[1] != -1:# 如果标注以"I-"开头，则当前标注为一个实体的中间或结束
            _type = tag.split('-')[1] # 当前实体的类型
            if _type == chunk[0]: # 如果当前实体类型与之前识别的实体类型相同，则将其添加到之前识别的实体中
                chunk[2] = indx
            if indx == len(seq) - 1: # 如果当前标注是最后一个标注，则将之前识别的实体存入chunks列表中
                chunks.append(chunk)
        else:# 如果标注不以"S-"，"B-"或"I-"开头，则当前标注
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq,id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = [] # 存储实体的列表
    chunk = [-1, -1, -1]# 存储正在识别的实体信息的列表，初始值为-1
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"): # 如果实体的开始
            if chunk[2] != -1: # 如果当前正在识别的实体信息不是初始状态，则将其加入
                chunks.append(chunk)
            chunk = [-1, -1, -1] # 重置实体信息
            chunk[1] = indx # 存储实体开始的位置
            chunk[0] = tag.split('-')[1]# 存储实体的类型
            chunk[2] = indx#存储实体结束的位置
            if indx == len(seq) - 1:# 如果是序列的最后一个标签，则将当前实体信息加入实体列表
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:# 如果是实体的中间部分，并且当前正在识别实体的开始位置不是初始状态
            _type = tag.split('-')[1]# 获取实体类型
            if _type == chunk[0]:# 如果实体类型和当前正在识别的实体类型相同
                chunk[2] = indx # 更新实体结束位置

            if indx == len(seq) - 1:# 如果是序列的最后一个标签，则将当前实体信息加入实体列表
                chunks.append(chunk)
        else:
            if chunk[2] != -1:# 如果当前正在识别的实体信息不是初始状态，则将其加入实体列表
                chunks.append(chunk)
            chunk = [-1, -1, -1]# 重置实体信息
    return chunks


def get_entities(seq,id2label,markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio','bios']
    if markup =='bio':
        return get_entity_bio(seq,id2label)
    else:
        return get_entity_bios(seq,id2label)