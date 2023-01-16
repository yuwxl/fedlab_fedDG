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
import threading
import time
from glob import glob

import numpy as np
import torch
from torch.multiprocessing import Queue

from utils.util import _connectivity_region_analysis, _eval_dice, _eval_haus
from ..network_manager import NetworkManager
from ..coordinator import Coordinator

from ...utils.message_code import MessageCode
from ...utils import Logger

DEFAULT_SERVER_RANK = 0


class ServerManager(NetworkManager):
    """Base class of ServerManager.

    Args:
        network (DistNetwork): network configuration.
        handler (ParameterServerBackendHandler): performe global server aggregation procedure.
    """
    def __init__(self, network, handler):
        super().__init__(network)
        self._handler = handler
        self.coordinator = None

    def setup(self):
        """Initialization Stage. 
            
        - Server accept local client num report from client manager.
        - Init a coordinator for client_id mapping.
        """
        super().setup()
        rank_client_id_map = {}

        for rank in range(1, self._network.world_size):
            _, _, content = self._network.recv(src=rank)
            rank_client_id_map[rank] = content[0].item()
        self.coordinator = Coordinator(rank_client_id_map)
        if self._handler is not None:
            self._handler.client_num_in_total = self.coordinator.total


class ServerSynchronousManager(ServerManager):
    """Synchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronously communicate with clients following agreements defined in :meth:`main_loop`.

    Args:
        network (DistNetwork): Manage ``torch.distributed`` network communication.
        handler (ParameterServerBackendHandler): Backend calculation handler for parameter server.
        logger (Logger, optional): object of :class:`Logger`.
    """
    def __init__(self, network, handler, logger=Logger()):
        super(ServerSynchronousManager, self).__init__(network, handler)
        self._LOGGER = logger

    def shutdown(self):
        """Shutdown stage."""
        self.shutdown_clients()
        super().shutdown()

    def main_loop(self):
        """Actions to perform in server when receiving a package from one client.

        Server transmits received package to backend computation handler for aggregation or others
        manipulations.

        Loop:
            1. activate clients for current training round.
            2. listen for message from clients -> transmit received parameters to server backend.

        Note:
            Communication agreements related: user can overwrite this function to customize
            communication agreements. This method is key component connecting behaviors of
            :class:`ParameterServerBackendHandler` and :class:`NetworkManager`.

        Raises:
            Exception: Unexpected :class:`MessageCode`.
        """
        while self._handler.stop_condition() is not True:
            activate = threading.Thread(target=self.activate_clients)
            activate.start()
            while True:
                sender, message_code, payload = self._network.recv()
                if message_code == MessageCode.ParameterUpdate:
                    model_parameters = payload[0]
                    if self._handler.add_model(sender, model_parameters):
                        # 保存一次模型
                        snapshot_path = "../../fed_output(DP)/"
                        if not os.path.exists(snapshot_path):
                            os.makedirs(snapshot_path)
                        if not os.path.exists(snapshot_path + '/model'):
                            os.makedirs(snapshot_path + '/model')
                        # print("fed_output",snapshot_path)
                        unseen_site_idx = 0
                        loca = time.strftime('%Y-%m-%d-%H-%M-%S')
                        client0_data_list = []
                        # 准备测试的数据，可不用，直接在测试代码里面跑
                        client0_data_list.append(glob('../../dataset/Fundus/client1/*.npy'))
                        # print("len for eval", len(client0_data_list))
                        # 满足文件名唯一性且符合文件名定义规范
                        name = str(loca)

                        ## evaluation
                        self._LOGGER.info("epoch {} testing , site {}".format(name, unseen_site_idx))
                        dice, dice_array, haus, haus_array = self.test(client0_data_list, self._handler.model)
                        self._LOGGER.info(("   OD dice is: {}, std is {}".format(dice[0], np.std(dice_array[:, 0]))))
                        self._LOGGER.info(("   OC dice is: {}, std is {}".format(dice[1], np.std(dice_array[:, 1]))))
                        self._LOGGER.info(("   OD haus is: {}, std is {}".format(haus[0], np.std(haus_array[:, 0]))))
                        self._LOGGER.info(("   OC haus is: {}, std is {}".format(haus[1], np.std(haus_array[:, 1]))))
                        ## save model
                        save_mode_path = os.path.join(snapshot_path + '/model', 'epoch_' + name + '.pth')
                        torch.save(self._handler.model.state_dict(), save_mode_path)
                        self._LOGGER.info("save model (DP) to {}".format(save_mode_path))
                        # logging.info("save model to {}".format(save_mode_path))

                        break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))


    def test(self,data_list, test_net):

        test_data_list = data_list[0]
        # print('hhhlen data list',len(test_data_list))
        dice_array = []
        haus_array = []

        for fid, filename in enumerate(test_data_list):
            data = np.load(filename)
            image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
            mask = np.expand_dims(data[..., 3:].transpose(2, 0, 1), axis=0)
            image = torch.from_numpy(image).float()

            logit, pred, _ = test_net(image)
            pred_y = pred.cpu().detach().numpy()
            pred_y[pred_y > 0.75] = 1
            pred_y[pred_y < 0.75] = 0

            pred_y_0 = pred_y[:, 0:1, ...]
            pred_y_1 = pred_y[:, 1:, ...]
            processed_pred_y_0 = _connectivity_region_analysis(pred_y_0)
            processed_pred_y_1 = _connectivity_region_analysis(pred_y_1)
            processed_pred_y = np.concatenate([processed_pred_y_0, processed_pred_y_1], axis=1)
            dice_subject = _eval_dice(mask, processed_pred_y)
            haus_subject = _eval_haus(mask, processed_pred_y)
            dice_array.append(dice_subject)
            haus_array.append(haus_subject)

        dice_array = np.array(dice_array)
        haus_array = np.array(haus_array)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        haus_avg = np.mean(haus_array, axis=0).tolist()
        return dice_avg, dice_array, haus_avg, haus_array

    def activate_clients(self):
        """Activate subset of clients to join in one FL round

        Manager will start a new thread to send activation package to chosen clients' process rank.
        The ranks of clients are obtained from :meth:`handler.sample_clients`.
        """
        clients_this_round = self._handler.sample_clients()
        self._LOGGER.info(
            "client id list for this FL round: {}".format(clients_this_round))

        for client_id in clients_this_round:
            rank = client_id + 1

            model_parameters = self._handler.model_parameters  # serialized model params
            self._network.send(content=model_parameters,
                               message_code=MessageCode.ParameterUpdate,
                               dst=rank)

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to each client with :attr:`MessageCode.Exit` to ask client to exit.

        Note:
            Communication agreements related: User can overwrite this function to define package
            for exiting information.
        """
        for rank in range(1, self._network.world_size):
            self._network.send(message_code=MessageCode.Exit, dst=rank)


class ServerAsynchronousManager(ServerManager):
    """Asynchronous communication network manager for server

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Asynchronously communicate with clients following agreements defined in :meth:`run`.

    Args:
        network (DistNetwork): Manage ``torch.distributed`` network communication.
        handler (ParameterServerBackendHandler, optional): Backend computation handler for parameter server.
        logger (Logger, optional): object of :class:`Logger`.
    """
    def __init__(self, network, handler, logger=Logger()):
        super(ServerAsynchronousManager, self).__init__(network, handler)
        self._LOGGER = logger

        self.message_queue = Queue()

    def shutdown(self):
        self.shutdown_clients()
        super().shutdown()

    def main_loop(self):
        """Communication agreements of asynchronous FL.

        - Server receive ParameterRequest from client. Send model parameter to client.
        - Server receive ParameterUpdate from client. Transmit parameters to queue waiting for aggregation.

        Raises:
            ValueError: invalid message code.
        """
        watching = threading.Thread(target=self.watching_queue)
        watching.start()
        # snapshot_path = "../../../fed_output/"
        # if not os.path.exists(snapshot_path):
        #     os.makedirs(snapshot_path)
        # if not os.path.exists(snapshot_path + '/model'):
        #     os.makedirs(snapshot_path + '/model')
        # print("fed_output",snapshot_path)
        # ## save model
        # save_mode_path = os.path.join(snapshot_path + '/model', 'epoch_' + str(self._handler.server_time) + '.pth')
        # torch.save(self._handler.model_parameters.state_dict(), save_mode_path)
        # logging.info("save model to {}".format(save_mode_path))

        while self._handler.stop_condition() is not True:
            sender, message_code, payload = self._network.recv()

            if message_code == MessageCode.ParameterRequest:
                model_parameters = self._handler.model_parameters
                content = [
                    model_parameters,
                    torch.Tensor(self._handler.server_time)
                ]
                self._LOGGER.info(
                    "Send model to rank {}, current server model time is {}".
                    format(sender, self._handler.server_time))
                self._network.send(content=content,
                                   message_code=MessageCode.ParameterUpdate,
                                   dst=sender)

            elif message_code == MessageCode.ParameterUpdate:
                self.message_queue.put((sender, message_code, payload))

            else:
                raise ValueError(
                    "Unexpected message code {}".format(message_code))



    def watching_queue(self):

        """Asynchronous communication maintain a message queue. A new thread will be started to run this function."""

        while self._handler.stop_condition() is not True:
            _, _, payload = self.message_queue.get()
            model_parameters = payload[0]
            model_time = payload[1]
            self._handler._update_model(model_parameters, model_time)


    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to clients with ``MessageCode.Exit``.
        """
        for rank in range(1, self._network.world_size):
            _, message_code, _ = self._network.recv(src=rank)
            if message_code == MessageCode.ParameterUpdate:
                self._network.recv(
                    src=rank
                )  # the next package is model request, which is ignored in shutdown stage.
            self._network.send(message_code=MessageCode.Exit, dst=rank)
