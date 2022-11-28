import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    #print(len(labels)) #2708
    #xit()
    classes = set(labels) #{'Probabilistic_Methods', 'Rule_Learning', 'Theory', 'Case_Based', 'Neural_Networks', 'Reinforcement_Learning', 'Genetic_Algorithms'}
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)} #{'Neural_Networks': array([1., 0., 0., 0., 0., 0., 0.]), 'Genetic_Algorithms':
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    #print(labels_onehot)
    #   [[0 0 0 ... 0 0 1]
    #  [0 0 0 ... 0 0 0]
    #  [0 0 1 ... 0 0 0]
    #  ...
    #  [1 0 0 ... 0 0 0]
    #  [0 0 0 ... 1 0 0]
    #  [0 0 0 ... 0 0 1]]
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    # content file的每一行的格式为: <paper_id> <word_attributes> <class_label>
    # 分别对应 0, 1:-1, -1
    # feature为第二列到倒数第二列，labels为最后一列
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    #print(idx_features_labels.shape)  #['31336' '0' '0' ... '0' '0' 'Neural_Networks'] 1,1435  (2708, 1435)
    # 储存为csr型稀疏矩阵
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
   # print(features.shape) # (2708, 1433)

    labels = encode_onehot(idx_features_labels[:, -1])
  #  print(len(labels)) #2708
   # exit()
    # build graph
    # # 根据前面的contents与这里的cites创建图，算出edges矩阵与adj矩阵 idx表示节点
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
   # print(len(idx)) #[  31336 1061127 1106406 ... 1128978  117328   24043] 2708

    idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered为直接从边表文件中直接读取的结果，是一个(edge_num, 2)的数组，每一行表示一条边两个端点的idx
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    #print(len(edges_unordered)) #[     35    1033]  ..

    # flatten：降维，返回一维数组
   # print(edges_unordered.flatten()) #[     35    1033      35 ...  853118  954315 1155073]
    #exit()
    # 边的edges_unordered中存储的是端点id，要将每一项的old id换成编号number
    # 在idx_map中以idx作为键查找得到对应节点的编号，reshape成与edges_unordered形状一样的数组 5429 重新构建map
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    # 根据coo矩阵性质，这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，
    # 所以先创建一个长度为edge_num的全1数组，每个1的填充位置就是一条边中两个端点的编号，
    # 即edges[:, 0], edges[:, 1]，矩阵的形状为(node_size, node_size)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    #   (163, 402)	1.0   (163, 659)	1.0  #(2708, 2708)

    # build symmetric adjacency matrix
    # 邻接矩阵转换为对称的无向图邻接矩阵:
    # 对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵
    # 将i->j与j->i中权重最大的那个, 作为无向图的节点i与节点j的边权.
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #print(adj.shape) #(2708, 2708)
   # print(features.shape) #(2708, 1433)
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    #https://blog.csdn.net/weixin_42067234/article/details/80247194    print(adj.todense()) 变成正常矩阵
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    # # 论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    # 对每一行求和
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
   # print(rowsum.shape) #(2708, 1) 1可以去掉
    # 求倒数
    r_inv = np.power(rowsum, -1).flatten()
   # print(r_inv.shape) #[0.05       0.05882353 0.04545455 ... 0.05555556 0.05263158 0.05263158]
    #exit()
    # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_inv[np.isinf(r_inv)] = 0.
    # 构建对角元素为r_inv的对角矩阵
    r_mat_inv = sp.diags(r_inv)  #  (0, 0)	0.05 ...
   # print(r_mat_inv.shape) #(2708, 2708)
    #exit()
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    mx = r_mat_inv.dot(mx)  #   (0, 1426)	0.05

    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

