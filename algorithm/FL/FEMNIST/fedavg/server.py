import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import numpy as np
import collections
import copy
import torch
import ti_torch
import logging
from args import parse_args

import pickle

global args
args = parse_args()


class Server:
    def __init__(self, int_model, float_model):
        super().__init__()
        self.int_model = int_model
        self.float_model = float_model

        self.selected_clients = []
        self.select_clients_acc_loss = []
        self.int_updates = []
        self.float_updates = []

    def set_server_param(self, checkpoint, clients):
        self.float_model.load_state_dict(checkpoint)
        for c in clients:
            c.model.load_state_dict(self.float_model.load_state_dict())

    def savecheckpoint(self, epoch, flag):
        dir_modify_here = 'light-60-0.8-1.25-'
        state = {
            'net': self.float_model.state_dict()
        }
        if epoch > 0 and flag > 1:
            os.remove('./checkpoint/' + dir_modify_here + str(epoch - 1) + '.pth')
        torch.save(state, './checkpoint/' + dir_modify_here + str(epoch) + '.pth')

    def select_clients(self, my_round, possible_clients, num_clients=20):

        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)

        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        num_samples = {c.id: c.num_train_samples for c in clients}
        return ids, num_samples

    def train_model(self, epoch, num_epochs=1, batch_size=5, clients=None, model_type='int'):

        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """

        int_model_dict = None if self.int_model is None else copy.deepcopy(self.int_model.state_dict())
        float_model_dict = None if self.float_model is None else copy.deepcopy(self.float_model.state_dict())

        for cids, c in enumerate(clients):
            weight_float_list = []
            grad_float_list = []

            if c.model_type == 'int':
                regime = c.model.regime
                for s in regime:
                    if s['epoch'] == epoch:
                        ti_torch.GRAD_BITWIDTH = s['gb']
                        logging.info('changing gradient bitwidth: %d', ti_torch.GRAD_BITWIDTH)
                        break
                c.model.load_state_dict(int_model_dict)
                loss, update = c.train(epoch, cids, num_epochs, batch_size, c.model_type)
                print("loss: ", loss)
                self.int_updates.append((c.num_train_samples, copy.deepcopy(update)))
            else:
                c.model.load_state_dict(float_model_dict)
                loss, update = c.train(epoch, cids, num_epochs, batch_size, c.model_type)

                self.float_updates.append((c.num_train_samples, copy.deepcopy(update)))

    def update_model(self, clients, model_type='int', epoch=0):

        if model_type == 'int':
            self.update_int_model()
        elif model_type == 'float':

            self.update_float_model(epoch, clients)


        else:
            self.update_hybrid_model()

    def update_int_model(self):
        """
        """
        base, layer_name_list, _ = self.update_int_model_temp()

        state_dict_agg = self.construct_dict(model_type='int', base=base, layer_name_list=layer_name_list)
        para = copy.deepcopy(self.int_model.state_dict())
        self.int_model.load_state_dict(state_dict_agg)
        para_agg = self.int_model.state_dict()
        self.int_updates = []

    def update_int_model_temp(self):
        total_weight = 0.
        float_weight_list = []
        layer_name_list = []
        for (client_samples, client_model) in self.int_updates:
            weight = []
            weight_exp = []
            float_weight = []
            total_weight += client_samples
            layer_name_list, layer_val_list = list(client_model.keys()), list(client_model.values())
            for i in range(0, len(layer_val_list), 2):
                layer_int = (layer_val_list[i], layer_val_list[i + 1])
                layer_float = ti_torch.TiInt8ToFloat(layer_int)
                float_weight.append(float(client_samples) * layer_float)
            float_weight_list.append(float_weight)

        base = [0] * len(float_weight_list[0])
        for i in range(len(float_weight_list)):
            for j, v in enumerate(float_weight_list[i]):
                base[j] += v / total_weight

        return base, layer_name_list, total_weight

    def update_SGD_float_model(self, epoch):
        total_weight = 0.
        base = [0] * len(self.float_updates[0][1])
        layer_name_list = []
        for (client_samples, client_model) in self.float_updates:

            total_weight += client_samples

            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.float())

        avg_base = [v / total_weight for v in base]

        server_para = copy.deepcopy(self.float_model.state_dict())

        layer_name_list = list(server_para.keys())

        state_dict_agg = self.construct_dict(model_type='float', base=avg_base, layer_name_list=layer_name_list)

        layer_index = 0
        for float_weight in server_para.values():
            state_dict_agg[layer_name_list[layer_index]] += float_weight
            layer_index += 1

        self.float_model.load_state_dict(state_dict_agg)

        self.float_updates = []
        return total_weight, layer_name_list

    def update_float_model(self, epoch, clients):

        total_weight = 0.
        base = [0] * len(self.float_updates[0][1])
        layer_name_list = []
        for (client_samples, client_model) in self.float_updates:
            layer_name_list = list(client_model.keys())
            total_weight += client_samples
            for i, v in enumerate(list(client_model.values())):
                base[i] += (client_samples * v.float())

        avg_base = [v / total_weight for v in base]

        state_dict_agg = self.construct_dict(model_type='float', base=avg_base, layer_name_list=layer_name_list)
        para = copy.deepcopy(self.float_model.state_dict())

        self.float_model.load_state_dict(state_dict_agg)

        self.float_updates = []

        for c in clients:
            if c.model_type == 'int':
                c.model.load_state_dict(self.int_model.state_dict())
            elif c.model_type == 'float':
                c.model.load_state_dict(self.float_model.state_dict())
        return total_weight, layer_name_list

    def update_float_model_clip(self, epoch):
        total_weight = 0.
        base = [0] * len(self.float_updates[0][1])
        base_cpu = [0] * len(self.float_updates[0][1])
        layer_name_list = []

        if epoch > 0:
            pickle_file = open("../niti/weights/normal/300_0.99_3/pickle" + str(epoch - 1) + ".pkl", "rb")

            last_server_weight = pickle.load(pickle_file)

        for (client_samples, client_model) in self.float_updates:

            layer_name_list = list(client_model.keys())
            client_model_value = list(client_model.values())

            if epoch > 0:
                client_model_value = self.limit_weight_change(client_model_value, last_server_weight, epoch)
                client_model_value = client_model_value.tolist()

            total_weight += client_samples

            if epoch == 0:
                for i, v in enumerate(client_model_value):
                    base[i] += (client_samples * v.float())
                    base_cpu[i] += (client_samples * v.float()).data.cpu().numpy()
            elif epoch > 0:
                for i, v in enumerate(client_model_value):
                    base_cpu[i] += (client_samples * v.astype(np.float32))
                    base[i] += torch.tensor(client_samples * v.astype(np.float32)).cuda()

        avg_base = [v / total_weight for v in base]
        avg_base_cpu = [v / total_weight for v in base_cpu]

        pickle_file = open("../niti/weights/normal/300_0.99_3/pickle" + str(epoch) + ".pkl", "wb")

        pickle.dump(avg_base_cpu, pickle_file)
        pickle_file.close()

        state_dict_agg = self.construct_dict(model_type='float', base=avg_base, layer_name_list=layer_name_list)
        para = copy.deepcopy(self.float_model.state_dict())
        self.float_model.load_state_dict(state_dict_agg)
        self.float_updates = []

        return total_weight, layer_name_list

    def limit_weight_change(self, client_model, server_weight, num):

        clip_list = []

        for c_weight, s_weight in zip(client_model, server_weight):

            c_weight = c_weight.data.cpu().numpy()

            change = c_weight - s_weight

            l2 = np.linalg.norm(change)

            clip_init = 0.08

            if num <= 300:
                clip_norm = clip_init

            elif num > 300 and num <= 800:
                clip_norm = clip_init * 0.99 ** (num - 300)
            elif num > 800:
                clip_norm = clip_init * (0.99 ** 500) * 0.98 ** (num - 800)

            clip_weight_norm = change * clip_norm / max(l2, clip_norm)

            clip_list.append(clip_weight_norm)

        server_weight = np.array(server_weight)

        clip_list = np.array(clip_list)

        client_model = server_weight + clip_list

        return client_model

    def update_hybrid_model(self):

        total_float_weight, float_layer_name = self.update_float_model()
        int_base, int_layer_name, total_int_weight = self.update_int_model_temp()
        float_model_dict = list(self.float_model.state_dict().values())
        float_base = float_model_dict[::2]

        total_weight = total_int_weight + total_float_weight
        base = [0] * len(int_base)

        for i in range(len(int_base)):
            base[i] = (total_float_weight * float_base[i].float() + total_int_weight * int_base[
                i].float()) / total_weight

        int_model_avg = base
        state_dict_agg_int = self.construct_dict(model_type='int', base=int_model_avg, layer_name_list=int_layer_name)
        para = copy.deepcopy(self.int_model.state_dict())
        self.int_model.load_state_dict(state_dict_agg_int)
        para_agg = self.int_model.state_dict()
        self.int_updates = []

        float_model_avg = [0] * len(float_model_dict)
        for i in range(len(float_model_dict)):
            if i % 2 == 0:
                float_model_avg[i] = base[int(i / 2)]
            else:
                float_model_avg[i] = float_model_dict[i]
        state_dict_agg_float = self.construct_dict(model_type='float', base=float_model_avg,
                                                   layer_name_list=float_layer_name)
        para = copy.deepcopy(self.float_model.state_dict())
        self.float_model.load_state_dict(state_dict_agg_float)
        self.float_updates = []

    def construct_dict(self, model_type, base, layer_name_list):
        state_dict_agg = collections.OrderedDict()
        if model_type == 'int':
            layer_index = 0
            for float_weight in base:
                round_val, act_exp = ti_torch.weight_quant(float_weight)
                state_dict_agg[layer_name_list[layer_index]] = round_val
                state_dict_agg[layer_name_list[layer_index + 1]] = act_exp
                layer_index += 2
        else:
            layer_index = 0
            for weight in base:
                state_dict_agg[layer_name_list[layer_index]] = weight
                layer_index += 1

        return state_dict_agg

    def test_model(self, clients_to_test, model_type='int', set_to_use='test'):
        """
        """
        metrics = {}

        for c in clients_to_test:
            if c.model_type == 'int':
                c.model.load_state_dict(self.int_model.state_dict())
            elif c.model_type == 'float':
                c.model.load_state_dict(self.float_model.state_dict())

            if set_to_use == 'train':

                top1 = c.test(set_to_use='train')


            elif set_to_use == 'test':
                top1 = c.test(set_to_use='test')

            metrics[c.id] = top1
        return metrics


GPU = 'cuda:0'

BITWIDTH = 7
