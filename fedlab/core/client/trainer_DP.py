# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import sys
import time
from collections import OrderedDict
from glob import glob

from tensorboardX import SummaryWriter
import numpy as np
import torch

from tqdm import tqdm

from utils.losses import dice_loss
from ..client import ORDINARY_TRAINER
from ...utils import Logger
from ...utils.dp_mechanism import Gaussian_Simple, cal_sensitivity,Laplace
from ...utils.serialization import SerializationTool
from ..model_maintainer import ModelMaintainer

class ClientTrainer(ModelMaintainer):
    """An abstract class representing a client backend trainer.

    In our framework, we define the backend of client trainer show manage its local model.
    It should have a function to update its model called :meth:`train`.

    If you use our framework to define the activities of client, please make sure that your self-defined class
    should subclass it. All subclasses should overwrite :meth:`train`.

    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool): Use GPUs or not.
    """

    def __init__(self, model, cuda):
        super().__init__(model, cuda)
        self.client_num = 1  # default is 1.
        self.type = ORDINARY_TRAINER

    def train(self):
        """Override this method to define the algorithm of training your model. This function should manipulate :attr:`self._model`"""
        raise NotImplementedError()

    def evaluate(self):
        """Evaluate quality of local model."""
        raise NotImplementedError()


class ClientContTrainer(ClientTrainer):
    # 对比损失训练
    def __init__(self,
                 model,
                 data_loader,
                 epochs,
                 optimizer,
                 criterion,
                 meta_step_size,
                 clip_value,
                 base_lr,
                 client_idx,
                 dp_mechanism='no_dp',
                 dp_epsilon=20,
                 dp_delta=1e-5,
                 dp_clip=20,
                 cuda=True,
                 logger=Logger()):
        super(ClientContTrainer, self).__init__(model, cuda)

        self._data_loader = data_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.meta_step_size = meta_step_size
        self.clip_value = clip_value
        self.base_lr = base_lr
        self.client_idx = client_idx
        self.dp_mechanism = dp_mechanism
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_clip = dp_clip
        self._LOGGER = logger
        self.device = torch.device(
            'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    def extract_contour_embedding(self,contour_list, embeddings):

        average_embeddings_list = []
        for contour in contour_list:
            contour_embeddings = contour * embeddings
            average_embeddings = torch.sum(contour_embeddings, (-1, -2)) / torch.sum(contour, (-1, -2))
            average_embeddings_list.append(average_embeddings)
        return average_embeddings_list

    def clip_gradients(self, net):
        if self.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            for k, v in net.named_parameters():
                v.grad /= max(1, v.grad.norm(1) / self.dp_clip)
        elif self.dp_mechanism == 'Gaussian':
            # Gaussian use 2 norm
            for k, v in net.named_parameters():
                v.grad /= max(1, v.grad.norm(2) / self.dp_clip)

    def add_noise(self, net):
        sensitivity = cal_sensitivity(self.base_lr, self.dp_clip, len(self._data_loader))
        if self.dp_mechanism == 'Laplace':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Laplace(epsilon=self.dp_epsilon, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.device)
                    v += noise
        elif self.dp_mechanism == 'Gaussian':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.device)
                    v += noise

    def train(self, model_parameters) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure (DP) is running")
        writer = SummaryWriter('../../fed_output(DP)/event/client_event{}/'.format(self.client_idx))
        self._model.train()
        iter_num = 0
        for i_batch, sampled_batch in tqdm(enumerate(self._data_loader),total =len(self._data_loader)-1,leave = False,colour = 'blue',ncols=50):
            time.sleep(0.001)
            volume_batch, label_batch, disc_contour, disc_bg, cup_contour, cup_bg = sampled_batch['image'], \
                                                                                    sampled_batch['label'], \
                                                                                    sampled_batch['disc_contour'], \
                                                                                    sampled_batch['disc_bg'], \
                                                                                    sampled_batch['cup_contour'], \
                                                                                    sampled_batch['cup_bg']
            volume_batch_raw_np = volume_batch[:, :3, ...]
            volume_batch_trs_1_np = volume_batch[:, 3:6, ...]
            # 因为没有 cuda所以修改了
            if self.cuda:
                volume_batch_raw, volume_batch_trs_1, label_batch = \
                    volume_batch_raw_np.cuda(), volume_batch_trs_1_np.cuda(), label_batch.cuda()
            # 因为没有cuda 所以注释掉了
                disc_contour, disc_bg, cup_contour, cup_bg = disc_contour.cuda(), disc_bg.cuda(), cup_contour.cuda(), cup_bg.cuda()

            outputs_soft_inner, outputs_mask_inner, embedding_inner = self._model(volume_batch_raw)
            loss_inner = dice_loss(outputs_soft_inner, label_batch)
            grads = torch.autograd.grad(loss_inner, self._model.parameters(), retain_graph=True)

            # 元训练 中 得到快参数 fast weights
            fast_weights = OrderedDict(
                (name, param - torch.mul(self.meta_step_size, torch.clamp(grad, 0 - self.clip_value, self.clip_value))) for
                ((name, param), grad) in
                zip(self._model.named_parameters(), grads))

            # outer loop evaluation
            outputs_soft_outer_1, outputs_mask_outer_1, embedding_outer = self._model(volume_batch_trs_1,
                                                                                      fast_weights)  # alpha
            loss_outer_1_dice = dice_loss(outputs_soft_outer_1, label_batch)

            inner_disc_ct_em, inner_disc_bg_em, inner_cup_ct_em, inner_cup_bg_em = \
                self.extract_contour_embedding([disc_contour, disc_bg, cup_contour, cup_bg], embedding_inner)
            outer_disc_ct_em, outer_disc_bg_em, outer_cup_ct_em, outer_cup_bg_em = \
                self.extract_contour_embedding([disc_contour, disc_bg, cup_contour, cup_bg], embedding_outer)
            #  print("cup_contour.shape1.5", cup_contour.shape)
            disc_ct_em = torch.cat((inner_disc_ct_em, outer_disc_ct_em), 0)
            disc_bg_em = torch.cat((inner_disc_bg_em, outer_disc_bg_em), 0)
            cup_ct_em = torch.cat((inner_cup_ct_em, outer_cup_ct_em), 0)
            cup_bg_em = torch.cat((inner_cup_bg_em, outer_cup_bg_em), 0)
            disc_em = torch.cat((disc_ct_em, disc_bg_em), 0)
            cup_em = torch.cat((cup_ct_em, cup_bg_em), 0)
            label = np.concatenate([np.ones(disc_ct_em.shape[0]), np.zeros(disc_bg_em.shape[0])])
            label = torch.from_numpy(label)

            disc_cont_loss = self.criterion(disc_em, label)
            cup_cont_loss = self.criterion(cup_em, label)
            cont_loss = disc_cont_loss + cup_cont_loss
            loss_outer = loss_outer_1_dice + cont_loss * 0.1  # gamma 设为 0.1

            total_loss = loss_inner + loss_outer

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.dp_mechanism != 'no_dp':
                    print("Gaussian1!!")
                    self.clip_gradients(self._model)
            self.optimizer.step()

            iter_num = iter_num + 1
            if iter_num % 5 == 0:
                writer.add_scalar('lr', self.base_lr, iter_num)
                writer.add_scalar('loss/inner', loss_inner, iter_num)
                writer.add_scalar('loss/outer', loss_outer, iter_num)
                writer.add_scalar('loss/total', total_loss, iter_num)
                self._LOGGER.info(
                    'Epoch: [%d] client [%d] iteration [%d / %d] : inner loss : %f outer dice loss : %f outer cont loss : %f outer loss : %f total loss : %f' % \
                    (i_batch, self.client_idx, iter_num, len(self._data_loader), loss_inner.item(),
                     loss_outer_1_dice.item(), cont_loss.item(), loss_outer.item(), total_loss.item()))

            if iter_num % 20 == 0:
                image = np.array(volume_batch_raw_np[0, 0:3, :, :], dtype='uint8')
                writer.add_image('train/RawImage', image, iter_num)

                image = np.array(volume_batch_trs_1_np[0, 0:3, :, :], dtype='uint8')
                writer.add_image('train/TrsImage', image, iter_num)

                image = outputs_soft_inner[0, 0:1, ...].data.cpu().numpy()
                writer.add_image('train/RawDiskMask', image, iter_num)
                image = outputs_soft_inner[0, 1:, ...].data.cpu().numpy()
                writer.add_image('train/RawCupMask', image, iter_num)

                image = np.array(disc_contour[0, 0:1, :, :].data.cpu().numpy())  # , dtype='uint8')
                writer.add_image('train/disc_contour', image, iter_num)

                image = np.array(disc_bg[0, 0:1, :, :].data.cpu().numpy())  # , dtype='uint8')
                writer.add_image('train/disc_bg', image, iter_num)

                #                    #("cup_contour.shape2",cup_contour.shape)
                image = np.array(cup_contour[0, 0:1, :, :].data.cpu().numpy())  # , dtype='uint8')
                # print("image.shape",image.shape)
                writer.add_image('train/cup_contour', image, iter_num)

                image = np.array(cup_bg[0, 0:1, :, :].data.cpu().numpy())  # , dtype='uint8')
                writer.add_image('train/cup_bg', image, iter_num)
        self._LOGGER.info("Local train procedure is finished")

        # add noises to parameters
        if self.dp_mechanism != 'no_dp':
            print("Gaussian2!!")
            self.add_noise(self._model)

        return self.model_parameters


class ClientSGDTrainer(ClientTrainer):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): PyTorch model.
        data_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        epochs (int): the number of local epoch.
        optimizer (torch.optim.Optimizer, optional): optimizer for this client's model.
        criterion (torch.nn.Loss, optional): loss function used in local training process.
        cuda (bool, optional): use GPUs or not. Default: ``True``.
        logger (Logger, optional): :object of :class:`Logger`.
    """

    def __init__(self,
                 model,
                 data_loader,
                 epochs,
                 optimizer,
                 criterion,
                 cuda=True,
                 logger=Logger()):
        super(ClientSGDTrainer, self).__init__(model, cuda)

        self._data_loader = data_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self._LOGGER = logger

    def train(self, model_parameters) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            for inputs, labels in tqdm(self._data_loader,
                                       desc="{}, Epoch {}".format(self._LOGGER.name, ep)):
                if self.cuda:
                    inputs, labels = inputs.cuda(self.gpu), labels.cuda(
                        self.gpu)

                outputs = self._model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")
        return self.model_parameters
