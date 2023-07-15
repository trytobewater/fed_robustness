import json
import numpy as np
import os
from collections import defaultdict
#coding=utf-8

def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_cifar_dir(data_dir, flag):

    groups = []

    data = defaultdict(lambda: None)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]

    for f in files:

        f_name = f.split("_")[1]
        client = f_name.split('.')[0]

        # 选择要更改的clients
        if client == str(flag):

            file_path = os.path.join(data_dir, f)
            # print(file_path)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            if 'hierarchies' in cdata:
                groups.extend(cdata)

            # data.update({str(client): cdata})
            data.update(cdata)

            return client, groups, data, file_path


def read_data(train_data_dir, test_data_dir, flag):

    train_clients, train_groups, train_data, train_file_path = read_cifar_dir(train_data_dir, flag)
    test_clients, test_groups, test_data, test_file_path = read_cifar_dir(test_data_dir, flag)

    # print('train clients:' + str(train_clients))
    # print('test clients:' + str(test_clients))

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data, train_file_path, test_file_path
