import os
import os.path as osp
import random
import csv
import json
import gzip


def save_data(task, split, data, save_dir, num_train=None, client_id=None):
    if num_train is None:
        num_train = int(0.9 * len(data))
    if client_id is not None:
        save_dir = osp.join(save_dir, str(client_id))
    else:
        save_dir = osp.join(save_dir, task)
    print('save {} {} file to \'{}\''.format(task, split, save_dir))
    os.makedirs(save_dir, exist_ok=True)

    if split == 'train':
        train_data = data[:num_train]
        val_data = data[num_train:]
        with open(osp.join(save_dir, 'train.json'), 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        with open(osp.join(save_dir, 'val.json'), 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
    elif split in {'dev', 'test'}:
        with open(osp.join(save_dir, 'test.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def load_data(task, root, split, save_dir, num_train=None, num_val=None, num_test=None, num_clients=None, client_id=None):
    data = []
    if task == 'imdb':
        data = []
        pos_files = os.listdir(osp.join(root, split, 'pos'))
        neg_files = os.listdir(osp.join(root, split, 'neg'))
        for file in pos_files:
            path = osp.join(root, split, 'pos', file)
            with open(path) as f:
                line = f.readline()
            data.append({'text': line, 'label': 1})
        for file in neg_files:
            path = osp.join(root, split, 'neg', file)
            with open(path) as f:
                line = f.readline()
            data.append({'text': line, 'label': 0})
        random.shuffle(data)

    elif task == 'agnews':
        with open(osp.join(root, split + '.csv'), encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
            for i, row in enumerate(csv_reader):
                label, title, description = row
                label = int(label) - 1
                text = ' [SEP] '.join((title, description))
                data.append({'text': text, 'label': label})

    elif task == 'squad':
        with open(osp.join(root, split + '.json'), 'r', encoding='utf-8') as reader:
            raw_data = json.load(reader)['data']
        for line in raw_data:
            for para in line['paragraphs']:
                context = para['context']
                for qa in para['qas']:
                    data.append({'context': context, 'qa': qa})

    elif task == 'newsqa':
        with gzip.GzipFile(osp.join(root, split + '.jsonl.gz'), 'r') as reader:
            content = reader.read().decode('utf-8').strip().split('\n')[1:]
            raw_data = [json.loads(line) for line in content]
        for line in raw_data:
            context = line['context']
            for qa in line['qas']:
                data.append({'context': context, 'qa': qa})

    elif task in {'cnndm', 'msqg'}:
        src_file = osp.join(root, split + '.src')
        tgt_file = osp.join(root, split + '.tgt')
        with open(src_file) as f:
            src_data = [line.strip().replace('<S_SEP>', '[SEP]') for line in f]
        with open(tgt_file) as f:
            tgt_data = [line.strip().replace('<S_SEP>', '[SEP]') for line in f]
        for src, tgt in zip(src_data, tgt_data):
            data.append({'src': src, 'tgt': tgt})

    # if num_train is None:
    #     save_data(task, split, data, save_dir)

    # split each dataset and distribute to multiple clients
    uniform_split = True
    if split == 'train':
        if num_train and num_val:
            uniform_split = False
            num_split = num_train + num_val
    elif split in {'dev', 'test'}:
        if num_test:
            uniform_split = False
            num_split = num_test

    print('Task: {} ({})'.format(task, split))
    data_i = 0
    for i in range(num_clients[task]):
        if uniform_split:
            n = len(data) // num_clients[task]
            num_split = n if i < num_clients[task] - 1 else len(data) - n * (num_clients[task] - 1)

        cur_client_id = client_id + i
        if data_i + num_split <= len(data):
            cur_data = data[data_i: data_i + num_split]
            data_i += num_split
        else:
            cur_data = data[data_i:]
            num_add = num_split - len(cur_data)
            cur_data += data[:num_add]
            data_i = num_add

        print('Client id: {}\tNum samples: {}'.format(i, num_split))
        save_data(task, split, cur_data, save_dir, num_train=num_train, client_id=cur_client_id)

    return data


def main():
    seed = 123
    random.seed(seed)
    data_root = {'imdb': '/mnt/dongchenhe.dch/datasets/imdb/',
                 'agnews': '/mnt/dongchenhe.dch/datasets/agnews/',
                 'squad': '/mnt/dongchenhe.dch/datasets/squad2.0/',
                 'newsqa': '/mnt/dongchenhe.dch/datasets/newsqa/',
                 'cnndm': '/mnt/dongchenhe.dch/datasets/glge/cnndm/',
                 'msqg': '/mnt/dongchenhe.dch/datasets/glge/msqg/'}
    save_dir = '/mnt/dongchenhe.dch/datasets/fednlp/v5/'
    num_clients = {'imdb': 1,
                   'agnews': 3,
                   'squad': 3,
                   'newsqa': 2,
                   'cnndm': 5,
                   'msqg': 4}

    client_id = 1
    for task, root in data_root.items():
        load_data(task, root, 'train', save_dir,
                  num_clients=num_clients,
                  client_id=client_id)
        if task in {'squad', 'newsqa', 'cnndm', 'msqg'}:
            load_data(task, root, 'dev', save_dir,
                      num_clients=num_clients,
                      client_id=client_id)
        elif task in {'imdb', 'agnews'}:
            load_data(task, root, 'test', save_dir,
                      num_clients=num_clients,
                      client_id=client_id)
        client_id += num_clients[task]


if __name__ == '__main__':
    main()
