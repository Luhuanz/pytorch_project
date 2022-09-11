from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
import torch

class En_DeDataSet(Dataset):

    def __init__(self, embedding, label):
        """

        :param embedding: 由wordEmbedding类产生的embedding
        :param label:对应于wordEmbedding的标签
        """

        self.embedding_Tensor = embedding
        self.label_Tensor = label

    def __len__(self):  # 自定义dataset必须
        return self.embedding_Tensor.size()[0]

    def __getitem__(self, idx):# 自定义dataset必须

        sample = {"text": self.embedding_Tensor[idx], "label": self.label_Tensor[idx]}
        return sample

class gpu_DataLoader(Module):

    def __init__(self, device, batch_size=4,num_workers=4,shuffle= True):
        super(gpu_DataLoader, self).__init__()
        self.device = device
        self.batch_size = batch_size #4
        self.num_workers = num_workers #4
        self.shuffle = shuffle

    def forward(self, dataset):

        dataloader = DataLoader(dataset= dataset, batch_size= self.batch_size,
                                num_workers= self.num_workers, shuffle= self.shuffle)

        for data_label in dataloader:
            yield data_label["text"].to(self.device), data_label["label"].to(self.device)

if __name__ == '__main__':

    from Source import DataSource
    from Source import text_paser
    from TokenTensorizer import TokenTensorizer

    test_data_path = {"text_filepath": "./iwslt2016/de-en",
                      "text_filename": ["IWSLT16.TED.dev2010.de-en.en.xml"],
                      "label_filepath": "./iwslt2016/de-en",
                      "label_filename": ["IWSLT16.TED.dev2010.de-en.de.xml"]}
    train_data_path = {"text_filepath": "./iwslt2016/de-en",
                       "text_filename": ["train.tags.de-en.en"],
                       "label_filepath": "./iwslt2016/de-en",
                       "label_filename": ["train.tags.de-en.de"]}
    # 加载数据
    datasource = DataSource(train_data_path)
    data = datasource(text_paser)

    # 得到wordEmbedding
    embedder = TokenTensorizer(num_embeddings=len(list(data["text_word_to_indexer"].keys())), embedding_dim=100,
                                    max_len=20, pretrain_path=None)
    embedding, label = embedder(data)
    print(embedding[0].size())
    print(label[0].size())

    dataset = En_DeDataSet(embedding= embedding, label= label)

    device = torch.device(0 if torch.cuda.is_available() else -1)

    gpu_dataloader = gpu_DataLoader(device= device)

    batch_data = gpu_dataloader(dataset)
    # print(batch_data)
    # count=0
    # for text, label in batch_data:  #196,884/4  49221
    #     print(text.shape,label.shape)
    #     count+=1
    # print(count)