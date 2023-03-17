import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_pickle(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)


def load_lines(path, encoding='utf8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = [line.strip() for line in f.readlines()]
        return lines


def write_lines(lines, path, encoding='utf8'):
    with open(path, 'w', encoding=encoding) as f:
        for line in lines:
            f.writelines('{}\n'.format(line))
