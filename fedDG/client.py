
import argparse
import logging
import random
import sys

import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import os
from glob import  glob
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader

sys.path.append("../")

from fedlab.core.client.manager import ClientPassiveManager
# from fedlab.core.client.trainer import ClientContTrainer
from fedlab.core.client.trainer_DP import ClientContTrainer
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
from fedlab.utils.dataset.sampler import RawPartitionSampler
from dataloaders.fundus_dataloader import Dataset,ToTensor
from networks.unet2d import Unet2D
parser = argparse.ArgumentParser(description="Distbelief training example")

# python client.py --ip 127.0.0.1 --port 3002 --world_size 3 --rank 1 --client_idx 1
parser.add_argument("--ip", type=str,default='127.0.0.1')
parser.add_argument("--port", type=str,default='3002')
parser.add_argument("--world_size", type=int,default=4)
parser.add_argument("--rank", type=int,default=1)
parser.add_argument('--client_idx', type=int, default=1, help='which client to train')
parser.add_argument('--dp_mechanism', type=str, default='Gaussian',help='differential privacy mechanism')
parser.add_argument('--dp_epsilon', type=float, default=10,help='differential privacy epsilon')
parser.add_argument('--dp_delta', type=float, default=1e-5,help='differential privacy delta')
parser.add_argument('--dp_clip', type=float, default=10,help='differential privacy clip')
parser.add_argument('--lr', type=float, default=0.01, help="DP learning rate")
parser.add_argument('--max_epoch', type=int,  default=1, help='maximum epoch number to train')
parser.add_argument("--ethernet", type=str, default=None)
parser.add_argument("--cuda", type=bool, default=True)

parser.add_argument('--batch_size', type=int, default=5, help='batch_size per gpu')
parser.add_argument('--clip_value', type=float,  default=100, help='maximum epoch number to train')
parser.add_argument('--meta_step_size', type=float,  default=1e-3, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--unseen_site', type=int, default=0, help='batch_size per gpu')
args = parser.parse_args()


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# batch_size = args.batch_size * len(args.gpu.split(','))
batch_size = args.batch_size
base_lr = args.base_lr
client_idx = args.client_idx
max_epoch = args.max_epoch

slice_num = np.array([50, 99, 320, 320])
volume_size = [384, 384, 3]
unseen_site_idx = args.unseen_site
source_site_idx = [0, 1, 2, 3]
source_site_idx.remove(unseen_site_idx)

# 加载训练数据

freq_site_idx = source_site_idx.copy()      # 源域初始为 0 1 2 3 ==> 去掉默认测试域0，为 1 2 3
if client_idx != unseen_site_idx:       #不可见域，即留一域 初始为默认为 0
    freq_site_idx.remove(client_idx)        #如果当前客户域不是保留的测试域，将它从域集合中剔除
# print("freq_site_idx",freq_site_idx)        # 加载数据集

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)
dataset = Dataset(client_idx=client_idx, freq_site_idx=freq_site_idx,
                        split='train', transform = transforms.Compose([
                        ToTensor(),
                        ]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
model = Unet2D()
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
temperature = 0.05
cont_loss_func = losses.NTXentLoss(temperature)  # 对比损失

network = DistNetwork(
    address=(args.ip, args.port),
    world_size=args.world_size,
    rank=args.rank,
    ethernet=args.ethernet,
)

LOGGER = Logger(log_name='client{}_log.txt'.format(client_idx),log_file="../../fed_output(DP)/Log/")
trainer =  ClientContTrainer(
    model,
    dataloader,
    epochs=args.max_epoch,
    optimizer=optimizer,
    criterion=cont_loss_func,
    meta_step_size=args.meta_step_size,
    clip_value=args.clip_value,
    base_lr=args.base_lr,
    client_idx=args.client_idx,
    dp_mechanism=args.dp_mechanism,
    dp_epsilon = args.dp_epsilon,
    dp_delta = args.dp_delta,
    dp_clip = args.dp_clip,
    cuda=args.cuda,
    logger=LOGGER,
)
manager_ = ClientPassiveManager(trainer=trainer,
                                network=network,
                                logger=LOGGER)
manager_.run()
