import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):
#: 初始化了实验的超参数、VAE 模型以及其他相关变量。
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass'] #控制是否在第一次反向传播时保留计算图以供后续使用。
        except:
            pass
#调用了 VAE 模型的前向传播方法，将输入数据传递给模型并返回输出结果。
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
#定义了在训练集上进行一次训练的步骤，包括计算损失函数、执行反向传播、记录日志等操作。
    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device
#在每个训练步骤中如何计算损失并进行反向传播。
        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']
#在验证集上进行一次验证的步骤，同样包括计算损失函数、记录日志等操作。
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
#每个验证周期结束时生成样本并保存重建样本和随机生成样本的图像。
    def on_validation_end(self) -> None:
        self.sample_images()
    #这段代码是定义了一个生成样本的函数，包括重建样本和随机生成样本。
    def sample_images(self):
        # Get sample reconstruction image
        #self.trainer.datamodule.test_dataloader()返回测试集的数据加载器对象。
        #next(iter(...))用于从该数据加载器对象中获取一批数据，返回的是一个包含数据和标签的元组，即(data, label)。
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass
#: 它配置了优化器和学习率调度器，用于模型的参数更新。其中，优化器使用 Adam 算法，学习率调度器使用指数衰减。
    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
