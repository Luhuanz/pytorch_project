import os
import yaml #用于读取和解析YAML文件
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn #torch.backends.cudnn模块，用于设置cuDNN加速
from pytorch_lightning import Trainer#Trainer类，用于训练和测试PyTorch Lightning模型
from pytorch_lightning.loggers import TensorBoardLogger #，用于记录训练日志并可视化训练过程
from pytorch_lightning.utilities.seed import seed_everything #设置随机数种子，保证实验可重复
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  #回调函数，分别用于监控学习率和保存模型
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin #用于启用分布式训练

# PyTorch Lightning是一个开源的轻量级高级应用程序框架，用于训练PyTorch模型。
# 它提供了一些高级功能，如自动化训练循环、分布式训练、可插拔回调、可视化等，可以大大简化训练流程并提高代码可读性。
# 使用PyTorch Lightning，您可以将重点放在模型的开发和改进上，而不必担心训练和调试的复杂性。
# 另外，PyTorch Lightning还支持多种深度学习任务，如分类、目标检测、语言建模、生成对抗网络等。


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
#'configs/vae.yaml'
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml') #metavar='FILE'的目的是告诉用户需要指定一个文件路径作为参数。

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file) #yaml.safe_load()函数来解析YAML文件，并将解析结果存储在config变量中。
    except yaml.YAMLError as exc:
        print(exc)
#可以实现对训练过程中各种指标的实时记录和可视化，如损失函数、精度、学习率等。
# 该对象还支持分布式训练和多进程训练，并且可以与其他日志记录器集成，如WandB、MLFlow等。
#创建一个用于记录训练过程的TensorBoard日志对象。
tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
#这段代码使用了PyTorch Lightning中的seed_everything()函数，
seed_everything(config['exp_params']['manual_seed'], True)
# init 字典
#在函数定义时，*args 和 **kwargs 是用于接收不定数量的参数的特殊语法。
# 其中，*args 用于接收不定数量的位置参数（以元组的形式），而 **kwargs 用于接收不定数量的关键字参数（以字典的形式）。
#config 中的 "model_params" 字典中的 "name" 键指定的模型名称来创建相应的 VAE 模型实例
#config['model_params'] 中的所有键值对解包作为参数传递给模型构造函数。这个操作称为“字典解包”，它将字典中的键值对转换为函数参数。
#print(vae_models[config['model_params']['name']](**config['model_params']) ) #<class 'models.vanilla_vae.VanillaVAE'>
#config['model_params']['name'] 的值是 'VanillaVAE'，那么 vae_models['VanillaVAE'] 就会返回 models.vanilla_vae.VanillaVAE 这个类。

#print(config['model_params']['name']) #VanillaVAE

model = vae_models[config['model_params']['name']](**config['model_params']) #访问字典对象 config 中键值为 model_params 的键所对应的值，
experiment = VAEXperiment(model,
                          config['exp_params'])

data = VAEDataset(**config["data_params"])


data.setup()
#tb_logger TensorBoard 日志记录器
#LearningRateMonitor 会在每个 epoch 后记录当前的学习率，
# ModelCheckpoint 则会在每个 epoch 后保存模型的权重参数到指定的路径下
#的模型权重参数。具体来说，ModelCheckpoint 回调函数在每个 epoch 结束后会检查当前模型在验证集上的表现
# （即指定的 monitor 指标，这里是 val_loss）
# 是否优于之前保存的模型权重，如果是，则将当前模型权重保存到指定的路径（即指定的 dirpath 目录下）。
#strategy 参数设置了使用分布式训练（Distributed Data Parallel）的策略，并且关闭了未使用参数的检查
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)