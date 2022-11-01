import numpy as np
import pylab
from sklearn.manifold import TSNE



if __name__ == "__main__":
    X = np.loadtxt("cora.features")
    labels_data = np.loadtxt('cora_labels.txt', dtype=np.int)
    labels = np.zeros(labels_data.shape, dtype=np.int)
    for i in range(labels_data.shape[0]):
        labels[i] = labels_data[i, 1]
    tsne = TSNE(n_components=2, init='random')
    Y = tsne.fit_transform(X[:, 1:])    #后面两个参数分别是邻居数量以及投影的维度
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels_data[:, 1])
    pylab.show()
