import numpy as np
import pandas as pd
import torch
import pickle
import dgl
import argparse
from sklearn.neighbors import NearestNeighbors

def check_param_num(model):
    '''
    check num of model parameters

    :model: pytorch model object
    :return: int
    '''
    param_num = 0 
    for parameter in model.parameters():
        param_num += parameter.shape[0]
    return param

def node_to_item(nodes, id_dict, cateogry_dict):
    '''
    Transform node id to real item id

    :items: node id list
    :id_dict: {node id: item category id}
    :category_dict: {item category id: real item id}
    '''
    ids = [id_dict[i] for i in nodes]
    ids = [cateogry_dict[i] for i in ids]   
    return ids

def get_blocks(seeds, item_ntype, textset, sampler):
    blocks = []
    for seed in seeds:
        block = sampler.get_block(seed, item_ntype, textset)
        blocks.append(block)
    return blocks

def get_all_emb(gnn, seed_array, textset, item_ntype, neighbor_sampler, batch_size, device='cuda'):
    seeds = torch.arange(seed_array.shape[0]).split(batch_size)
    testset = get_blocks(seeds, item_ntype, textset, neighbor_sampler)

    gnn = gnn.to(device)
    gnn.eval()
    with torch.no_grad():
        h_item_batches = []
        for blocks in testset:
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)

            h_item_batches.append(gnn.get_repr(blocks))
        h_item = torch.cat(h_item_batches, 0)
    return h_item
    
def item_by_user_batch(graph, user_ntype, item_ntype, user_to_item_etype, weight, args):
    '''
    :return: list of interacted node ids by every users 
    '''
    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, weight, args.batch_size)

    graph_slice = graph.edge_type_subgraph([rec_engine.user_to_item_etype])
    n_users = graph.number_of_nodes(rec_engine.user_ntype)  # 유저개수
    latest_interactions = dgl.sampling.select_topk(graph_slice, args.k, rec_engine.timestamp, edge_dir='out')
    user, latest_items = latest_interactions.all_edges(form='uv', order='srcdst')
    # user, latest_items = (k * n_users)

    items_df = pd.DataFrame({'user': user.numpy(), 'item': latest_items.numpy()}).groupby('user')
    items_batch = [items_df.get_group(i)['item'].values for i in np.unique(user)]
    return items_batch

def prec(recommendations, ground_truth):
    n_users, n_items = ground_truth.shape
    K = recommendations.shape[1]
    user_idx = np.repeat(np.arange(n_users), K)
    item_idx = recommendations.flatten()
    relevance = ground_truth[user_idx, item_idx].reshape((n_users, K))
    hit = relevance.any(axis=1).mean()
    return hit

class LatestNNRecommender(object):
    def __init__(self, user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size):
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.user_to_item_etype = user_to_item_etype
        self.batch_size = batch_size
        self.timestamp = timestamp

    def recommend(self, full_graph, K, h_user, h_item):
        """
        Return a (n_user, K) matrix of recommended items for each user
        """
        graph_slice = full_graph.edge_type_subgraph([self.user_to_item_etype])
        n_users = full_graph.number_of_nodes(self.user_ntype)
        latest_interactions = dgl.sampling.select_topk(graph_slice, K, self.timestamp, edge_dir='out')
        user, latest_items = latest_interactions.all_edges(form='uv', order='srcdst')
        # each user should have at least one "latest" interaction
        assert torch.equal(user, torch.arange(n_users))

        recommended_batches = []
        user_batches = torch.arange(n_users).split(self.batch_size)
        for user_batch in user_batches:
            latest_item_batch = latest_items[user_batch]
            dist = h_item[latest_item_batch] @ h_item.t()

            # 기존 인터랙션 삭제
            # 이 부분을 주석처리했음
            # for i, u in enumerate(user_batch.tolist()):
            #     interacted_items = full_graph.successors(u, etype=self.user_to_item_etype)
            #     dist[i, interacted_items] = -np.inf
            recommended_batches.append(dist.topk(K, 1)[1])

        recommendations = torch.cat(recommended_batches, 0)
        return recommendations


def evaluate_nn(dataset, h_item, k, batch_size):
    g = dataset['train-graph']
    val_matrix = dataset['val-matrix'].tocsr()
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']
    timestamp = dataset['timestamp-edge-column']

    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size)

    recommendations = rec_engine.recommend(g, k, None, h_item).cpu().numpy()
    return prec(recommendations, val_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('item_embedding_path', type=str)
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    with open(args.item_embedding_path, 'rb') as f:
        emb = torch.FloatTensor(pickle.load(f))
    print(evaluate_nn(dataset, emb, args.k, args.batch_size))
