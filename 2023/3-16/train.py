import torch
#models的模块中导入用于命名实体识别（NER）的PyTorch模型
from models.ner_model import BertSoftmaxForNer, LEBertSoftmaxForNer, LEBertCrfForNer, BertCrfForNer
#argparse 模块提供了一种处理命令行参数的方式，可以让你的 Python 脚本在执行时更加灵活。使用 argparse
import argparse
#看起来你正在尝试从 torch.utils.tensorboard 模块中导入 SummaryWriter 类。
# 这个类是用于将训练和评估结果写入 TensorBoard 的 PyTorch 工具包中的一部分。
# TensorBoard 是一个用于可视化和理解训练过程和模型性能的工具。SummaryWriter 类可以用来创建 TensorBoard 摘要写入器，
# 并在训练过程中记录各种指标、参数和网络结构。
from torch.utils.tensorboard import SummaryWriter
import random
import os
import numpy as np
from os.path import join
#logger函数用于创建一个日志对象，可用于记录各种严重程度的消息，例如debug，info，warning，error和critical。
# loguru库是Python中流行的日志记录库，提供了更直观和Pythonic的日志记录方式。
from loguru import logger
import time
from transformers import BertTokenizer, BertConfig
from torch.utils.data import Dataset, DataLoader
from processors.processor import LEBertProcessor, BertProcessor
import json
from tqdm import tqdm
from metrics.ner_metrics import SeqEntityScore
import transformers


def set_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
  #  parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument("--output_path", type=str, default='output/', help='模型与预处理数据的存放位置')
    parser.add_argument("--pretrain_embed_path", type=str, default='/root/autodl-tmp/project/pretrain_model/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt', help='预训练词向量路径')

    parser.add_argument('--loss_type', default='ce', type=str, choices=['lsr', 'focal', 'ce'], help='损失函数类型')
    parser.add_argument('--add_layer', default=1, type=str, help='在bert的第几层后面融入词汇信息')
    parser.add_argument("--lr", type=float, default=1e-5, help='Bert的学习率')
    parser.add_argument("--crf_lr", default=1e-3, type=float, help="crf的学习率")
    parser.add_argument("--adapter_lr", default=1e-3, type=float, help="crf的学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--eps', default=1.0e-08, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size_train", type=int, default=4)
    parser.add_argument("--batch_size_eval", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=2, help="训练多少步，查看验证集的指标")
    parser.add_argument("--max_seq_len", type=int, default=150, help="输入的最大长度")
    parser.add_argument("--max_word_num", type=int, default=3, help="每个汉字最多融合多少个词汇信息")
    parser.add_argument("--max_scan_num", type=int, default=10000, help="取预训练词向量的前max_scan_num个构造字典树")
    parser.add_argument("--data_path", type=str, default="datasets/ner_data/resume/", help='数据集存放路径')
    # parser.add_argument("--train_file", type=str, default="datasets/cner/train.txt")
    # parser.add_argument("--dev_file", type=str, default="datasets/cner/dev.txt")
    # parser.add_argument("--test_file", type=str, default="datasets/cner/test.txt")
    parser.add_argument("--dataset_name", type=str, choices=['resume', "weibo", 'ontonote4', 'msra'], default='resume', help='数据集名称')
    parser.add_argument("--model_class", type=str, choices=['lebert-softmax', 'bert-softmax', 'bert-crf', 'lebert-crf'],
                        default='lebert-crf', help='模型类别')
    parser.add_argument("--pretrain_model_path", type=str, default="pretrain_model/bert-base-chinese")
    parser.add_argument("--overwrite", action='store_true', default=True, help="覆盖数据处理的结果")
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_eval", action='store_true', default=True)
    parser.add_argument("--load_word_embed", action='store_true', default=True, help='是否加载预训练的词向量')

    parser.add_argument('--markup', default='bios', type=str, choices=['bios', 'bio'], help='数据集的标注方式')
    parser.add_argument('--grad_acc_step', default=1, type=int, required=False, help='梯度积累的步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False, help='梯度裁剪阈值')
    parser.add_argument('--seed', type=int, default=42, help='设置随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    # parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping.early stop得到的模型的生成效果不一定会更好。")
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.')
   #它使用 argparse 库来解析命令行参数并将它们存储在 args 对象中。
    args = parser.parse_args()
    return args


def seed_everything(seed=42):
    """
    设置整个开发环境的seed
    :param seed:
    :return:
    """
    random.seed(seed)
# Python 代码片段，它将环境变量 PYTHONHASHSEED 的值设置为给定的 seed 值，用于控制 Python 的哈希种子，从而使得 Python 的哈希结果可预测。
#这将使得 Python 的哈希结果在不同的机器和不同的运行时环境下都是相同的，从而使得结果可预测。
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
#这是一个 PyTorch 代码片段，它将 PyTorch 的随机数生成器的行为设置为确定性的，以便在使用 GPU 加速时，获得相同的结果。
    torch.backends.cudnn.deterministic = True


def get_optimizer(model, args, warmup_steps, t_total):
    # todo 检查
#其中 no_bert 和 no_decay 是两个列表，用于在进行模型参数优化时，指定哪些参数不需要应用权重衰减和学习率衰减。
# 在深度学习中，通常需要对模型的参数进行优化，以获得更好的性能。优化算法通常需要对参数进行更新，而学习率衰减和权重衰减是两个常用的技巧，
# 用于控制参数更新的速度和规模。
# no_bert是一个列表，包含了模型中不需要进行学习率衰减的参数。这些参数通常是一些特殊的层或者模块，比如嵌入层、分类器和条件随机场等.\
# no_decay是一个列表，包含了模型中不需要进行权重衰减的参数。这些参数通常是偏置项、层归一化的偏置和缩放参数等。
    no_bert = ["word_embedding_adapter", "word_embeddings", "classifier",  "crf"]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

#用于配置不同参数的学习率和权重衰减。具体来说，optimizer_grouped_parameters 是一个列表，其中包含了四个字典，分别对应了四组参数，
# 它们的学习率和权重衰减系数都是不同的。
    optimizer_grouped_parameters = [
        # bert no_decay
      #将模型中的参数分为需要应用权重衰减的参数和不需要应用权重衰减的参数两组
    # 需要应用权重衰减的参数被设置为weight_decay=args.weight_decay，而不需要应用权重衰减的参数则被设置为weight_decay=0.0。
        {
            "params": [p for n, p in model.named_parameters() #获取到模型中所有参数的名称和值的元组
                       if (not any(nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and any(nd in n for nd in no_decay)],
          #如果当前参数不包含在no_bert列表中或者是Bert模型的词向量嵌入权重参数，则被认为是需要应用权重衰减的参数。 检查当前参数名称是否包含在no_decay列表中的任意一个元素，如果存在，则认为当前参数需要进行学习率更新。
            #如果no_decay列表中的任意一个元素出现在当前参数n的名称中，则说明当前参数不需要应用权重衰减，因此该分组中的这部分参数被设置为weight_decay=0.0。否则，该分组中的这部分参数需要应用权重衰减，被设置为weight_decay=args.weight_decay。
            "weight_decay": 0.0, 'lr': args.lr
        },
        # bert decay
        #权重衰减被应用于BERT的所有参数，除了不需要进行权重衰减的那些参数以外，
        {
            "params": [p for n, p in model.named_parameters()
                       if (not any(nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, 'lr': args.lr
        },
        # other no_decay
    #这一部分筛选的是需要应用权重衰减的自定义层的参数，并将不需要应用权重衰减的参数排除在外。
    # 这里的args.adapter_lr是自定义层的学习率超参数，这些参数的权重衰减被设置为0.0，因为通常情况下自定义层的参数不需要进行权重衰减。
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_bert) and n != 'bert.embeddings.word_embeddings.weight' and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, "lr": args.adapter_lr
        },
        # other decay
    #这里也筛选需要进行学习率衰减的自定义层的参数，将不需要进行学习率衰减的参数排除在外。这里的args.adapter_lr是自定义层的学习率超参数，
    # 这些参数的学习率衰减被设置为args.weight_decay，因为通常情况下自定义层的参数需要进行学习率衰减。
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_bert) and n != 'bert.embeddings.word_embeddings.weight' and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, "lr": args.adapter_lr
        }
    ]
# 定义优化器 optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
#args.eps是AdamW优化器中的epsilon值，用于增加数值稳定性。 optimizer_grouped_parameters是经过筛选和分组的优化器参数组 args.lr是学习率超参数
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)
#transformers.get_linear_schedule_with_warmup 是一个用于创建线性学习率调度程序的函数，其在训练的初始阶段采用一个小的学习率，
# 并在指定的 warm-up 步骤之后线性地增加学习率， 然后在训练的后期采用一个较小的学习率。这有助于防止训练期间出现梯度爆炸或梯度消失等问题，
# 从而提高训练的稳定性和效果。
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    return optimizer, scheduler


def train(model, train_loader, dev_loader, test_loader, optimizer, scheduler, args):
    logger.info("start training")
    model.train()
    device = args.device
    best = 0
    dev = 0
    for epoch in range(args.epochs):
        #使用logger记录日志
        logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            #一个batch包含输入的ids、token类型ids、注意力掩码以及标签ids
            step = epoch * len(train_loader) + batch_idx + 1
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label_ids = data['label_ids'].to(device)
            # 不同模型输入不同
            if args.model_class == 'bert-softmax':
                loss, logits = model(input_ids, attention_mask, token_type_ids, args.ignore_index, label_ids)
            elif args.model_class == 'bert-crf':
                loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
            elif args.model_class == 'lebert-softmax':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, args.ignore_index, label_ids)
            elif args.model_class == 'lebert-crf':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)

            loss = loss.mean()  # 对多卡的loss取平均

            # 梯度累积
            loss = loss / args.grad_acc_step
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 进行一定step的梯度累计之后，更新参数
            if step % args.grad_acc_step == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            # 评测验证集和测试集上的指标
            if step % args.eval_step == 0:
                logger.info('evaluate dev set')
                dev_result = evaluate(args, model, dev_loader)
                logger.info('evaluate test set')
                test_result = evaluate(args, model, test_loader)

# 使用了TensorBoard来记录模型的验证损失和F1得分，方便我们对模型性能的监控和比较。
 # 其中writer是TensorBoard的一个实例，add_scalar方法会在TensorBoard上绘制一个曲线图来展示数据的变化。
# 第一个参数是曲线的名称，第二个参数是要记录的数据，第三个参数是记录数据时的步数。
# 在训练过程中，这两行代码会在每个epoch结束时记录一次验证集的损失和F1得分。
                writer.add_scalar('dev loss', dev_result['loss'], step)
                writer.add_scalar('dev f1', dev_result['f1'], step)
                writer.add_scalar('dev precision', dev_result['acc'], step)
                writer.add_scalar('dev recall', dev_result['recall'], step)

                writer.add_scalar('test loss', test_result['loss'], step)
                writer.add_scalar('test f1', test_result['f1'], step)
                writer.add_scalar('test precision', test_result['acc'], step)
                writer.add_scalar('test recall', test_result['recall'], step)

                model.train()
                if best < test_result['f1']:
                    best = test_result['f1']
                    dev = dev_result['f1']
                    logger.info('higher f1 of testset is {}, dev is {} in step {} epoch {}'.format(best, dev, step, epoch+1))
                    # save_path = join(args.output_path, 'checkpoint-{}'.format(step))
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.output_path)
    logger.info('best f1 of test is {}, dev is {}'.format(best, dev))


def evaluate(args, model, dataloader):
    """
    计算数据集上的指标
    :param args:
    :param model:
    :param dataloader:
    :return:
    """
    model.eval()
    device = args.device
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    # Eval!
    logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = {}".format(len(dataloader)))
    # logger.info("  Batch size = {}".format(args.batch_size_eval))
    eval_loss = 0.0  #
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label_ids = data['label_ids'].to(device)
            # 不同模型输入不同
            if args.model_class == 'bert-softmax':
                loss, logits = model(input_ids, attention_mask, token_type_ids, args.ignore_index, label_ids)
            elif args.model_class == 'bert-crf':
                loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
            elif args.model_class == 'lebert-softmax':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, args.ignore_index,
                                     label_ids)
            elif args.model_class == 'lebert-crf':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)
            loss = loss.mean()  # 对多卡的loss取平均
            eval_loss += loss

            input_lens = (torch.sum(input_ids != 0, dim=-1) - 2).tolist()   # 减去padding的[CLS]与[SEP]
            if args.model_class in ['lebert-crf', 'bert-crf']:
                preds = model.crf.decode(logits, attention_mask).squeeze(0)
                preds = preds[:, 1:].tolist()  # 减去padding的[CLS]
            else:
                preds = torch.argmax(logits, dim=2)[:, 1:].tolist()  # 减去padding的[CLS]
            label_ids = label_ids[:, 1:].tolist()   # 减去padding的[CLS]
            # preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
            # label_ids = label_ids.cpu().numpy().tolist()
            for i in range(len(label_ids)):
                input_len = input_lens[i]
                pred = preds[i][:input_len]
                label = label_ids[i][:input_len]
                metric.update(pred_paths=[pred], label_paths=[label])

    logger.info("\n")
    eval_loss = eval_loss / len(dataloader)
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results *****")
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********"%key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    return results


MODEL_CLASS = {
    'lebert-softmax': LEBertSoftmaxForNer,
    'lebert-crf': LEBertCrfForNer,
    'bert-softmax': BertSoftmaxForNer,
    'bert-crf': BertCrfForNer
}
PROCESSOR_CLASS = {
    'lebert-softmax': LEBertProcessor,
    'lebert-crf': LEBertProcessor,
    'bert-softmax': BertProcessor,
    'bert-crf': BertProcessor
}


def main(args):
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path, do_lower_case=True)
    # 数据处理器
    processor = PROCESSOR_CLASS[args.model_class](args, tokenizer)
    args.id2label = processor.label_vocab.idx2token
    args.ignore_index = processor.label_vocab.convert_token_to_id('[PAD]')
    # 初始化模型配置
    config = BertConfig.from_pretrained(args.pretrain_model_path)
    config.num_labels = processor.label_vocab.size
    config.loss_type = args.loss_type
    if args.model_class in ['lebert-softmax', 'lebert-crf']:
        config.add_layer = args.add_layer
        config.word_vocab_size = processor.word_embedding.shape[0]
        config.word_embed_dim = processor.word_embedding.shape[1]
    # 初始化模型
    model = MODEL_CLASS[args.model_class].from_pretrained(args.pretrain_model_path, config=config).to(args.device)
    # 初始化模型的词向量
    if args.model_class in ['lebert-softmax', 'lebert-crf'] and args.load_word_embed:
        logger.info('initialize word_embeddings with pretrained embedding')
#processor中构建好的词向量矩阵 word_embedding 转换为 PyTorch 的 Tensor 格式，
#然后复制给模型中的 word_embeddings 层的权重。model.word_embeddings.weight.data 获取 model 模型的词嵌入层的权重（此时还是 Numpy 数组格式），
# 然后使用 torch.from_numpy() 将 processor.word_embedding 转为 PyTorch Tensor 格式，最后使用 copy_() 方法将 Tensor 的值复制给模型的权重
        model.word_embeddings.weight.data.copy_(torch.from_numpy(processor.word_embedding))

    # 训练
    if args.do_train:
        # 加载数据集
        train_dataset = processor.get_train_data()
        # train_dataset = train_dataset[:8]
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.num_workers)
        dev_dataset = processor.get_dev_data()
        # dev_dataset = dev_dataset[:4]
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                    num_workers=args.num_workers)
        test_dataset = processor.get_test_data()
        # test_dataset = test_dataset[:4]
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                     num_workers=args.num_workers)
        t_total = len(train_dataloader) // args.grad_acc_step * args.epochs #表示总共需要迭代的次数
        warmup_steps = int(t_total * args.warmup_proportion)
        # optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
        # scheduler = transformers.get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        # )
        optimizer, scheduler = get_optimizer(model, args, warmup_steps, t_total)
        train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args)

    # 测试集上的指标
    if args.do_eval:
        # 加载验证集
        dev_dataset = processor.get_dev_data()
        # dev_dataset = dev_dataset[:4]
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                    num_workers=args.num_workers)
        # 加载测试集
        test_dataset = processor.get_test_data()
        # test_dataset = test_dataset[:4]
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                     num_workers=args.num_workers)
        model = MODEL_CLASS[args.model_class].from_pretrained(args.output_path, config=config).to(args.device)
        model.eval()

        result = evaluate(args, model, dev_dataloader)
        logger.info('devset precision:{}, recall:{}, f1:{}, loss:{}'.format(result['acc'], result['recall'], result['f1'], result['loss'].item()))
        # 测试集上的指标
        result = evaluate(args, model, test_dataloader)
        logger.info(
            'testset precision:{}, recall:{}, f1:{}, loss:{}'.format(result['acc'], result['recall'], result['f1'],
                                                                     result['loss'].item()))


if __name__ == '__main__':
    # 设置参数
    args = set_train_args()
    seed_everything(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")

    pretrain_model = 'mengzi' if 'mengzi' in args.pretrain_model_path else 'bert-base'
   #根据命令行参数和默认值构造输出路径，将训练模型的结果保存在该路径下。
    #如果args.load_word_embed为真，即加载了预训练的单词嵌入
    args.output_path = join(args.output_path, args.dataset_name, args.model_class, pretrain_model, 'load_word_embed' if args.load_word_embed else 'not_load_word_embed')
    args.train_file = join(args.data_path, 'train.json')
    args.dev_file = join(args.data_path, 'dev.json')
    args.test_file = join(args.data_path, 'test.json')
    args.label_path = join(args.data_path, 'labels.txt')
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.do_train:
        cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
        logger.info(args)
#它用于在TensorBoard中可视化训练过程中的各种指标，如损失函数、准确率、梯度等。
    # 通过向SummaryWriter中添加scalar、histogram、image等类型的数据，
# 可以在TensorBoard中查看这些数据在训练过程中的变化情况
        writer = SummaryWriter(args.output_path)
    main(args)
