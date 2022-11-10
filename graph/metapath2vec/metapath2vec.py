import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from reading_data import DataReader, Metapath2vecDataset
from model import SkipGramModel
from download import AminerDataset, CustomDataset


class Metapath2VecTrainer:
    def __init__(self, args):
        if args.aminer:
            dataset = AminerDataset(args.path)
        else:
            # 我们这里使用这种情况
            # 指定path: NetDBIS
            dataset = CustomDataset(args.path)
        # 读入metapath文件并处理
        self.data = DataReader(dataset, args.min_count, args.care_type)
        # 实现封装
        dataset = Metapath2vecDataset(self.data, args.window_size)
        # https://zhuanlan.zhihu.com/p/30385675
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate)

        # 模型参数用args赋值
        self.output_file_name = args.output_file
        # 一共多少个节点N
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr
        # emb_size: 一共多少个节点N; emb_dimension: 维度如128
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):

        for iteration in range(self.iterations):
            print("\n\n\nIteration: " + str(iteration + 1))
            # 优化方式
            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            # 优化方式和学习率等
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            # 按batch训练
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    # 损失函数值
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    # 反向传播，更新参数
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Metapath2vec")
    # parser.add_argument('--input_file', type=str, help="input_file")
    parser.add_argument('--aminer', action='store_true', help='Use AMiner dataset')
    # 输入文件
    parser.add_argument('--path', type=str, help="input_path", default='net_dbis/output_path.txt')
    # 输出文件
    parser.add_argument('--output_file', type=str, help='output_file', default='result.txt')
    # embedding维度
    parser.add_argument('--dim', default=128, type=int, help="embedding dimensions")
    # 窗口大小
    parser.add_argument('--window_size', default=7, type=int, help="context window size")
    # 迭代次数
    parser.add_argument('--iterations', default=5, type=int, help="iterations")
    # batch size
    parser.add_argument('--batch_size', default=50, type=int, help="batch size")
    # 0: metapath2vec; 1: metapath2vec++
    parser.add_argument('--care_type', default=0, type=int,
                        help="if 1, heterogeneous negative sampling, else normal negative sampling")
    # 学习率
    parser.add_argument('--initial_lr', default=0.025, type=float, help="learning rate")
    # skip2gram模型的词频
    parser.add_argument('--min_count', default=5, type=int, help="min count")
    # 资源核数
    parser.add_argument('--num_workers', default=16, type=int, help="number of workers")
    args = parser.parse_args(args=[])
    m2v = Metapath2VecTrainer(args)
    m2v.train()
