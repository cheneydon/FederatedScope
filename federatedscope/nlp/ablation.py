import re
import numpy as np


def main():
    mode = 'pretrain'
    num_clients = 6

    if mode == 'pretrain':
        path = '/mnt/dongchenhe.dch/FederatedScope/federatedscope/exp/pretrain/pfednlp/st_im_ag_sq_ne_cn/gradient/hierarchical/cosine/group_2/exp_print.log'
        orders = {k1: {k2: 0 for k2 in range(num_clients)} for k1 in range(num_clients)}
        with open(path) as f:
            for line in f:
                res = re.search(r'client_id2group: (.+)', line)
                if res is not None:
                    s = res.groups()[0][1:-1]
                    id2group = [int(c) for c in s.split(' ')]
                    for i in range(num_clients):
                        for j in range(num_clients):
                            if id2group[i] == id2group[j]:
                                orders[i][j] += 1
        for k in range(num_clients):
            orders[k] = dict(sorted(orders[k].items(), key=lambda item: item[1], reverse=True))

    else:
        path = '/mnt/dongchenhe.dch/FederatedScope/federatedscope/exp/pfednlp/st_im_ag_sq_ne_cn/topk_3/outside_weight_0.0/exp_print.log'
        client_id2topk = []
        with open(path) as f:
            for line in f:
                res = re.search(r'client_id2topk: (.+)', line)
                if res is not None:
                    s = ''.join(res.groups()[0].split(' '))[1:-1]
                    ids = []
                    stack = []
                    for i in range(len(s)):
                        if s[i] == ',': continue
                        if s[i] == '[':
                            stack.append(s[i])
                        elif s[i] == ']':
                            cur_ids = []
                            while stack[-1] != '[':
                                num = stack.pop()
                                cur_ids = [num] + cur_ids
                            ids.append(cur_ids)
                            stack.pop()
                        else:
                            num = 0
                            while '0' <= s[i] <= '9':
                                num = num * 10 + int(s[i])
                                i += 1
                            stack.append(num)
                    client_id2topk.append(ids)

        client_id2topk = np.array(client_id2topk)
        orders = {k1: {k2: 0 for k2 in range(num_clients)} for k1 in range(num_clients)}
        for i in range(len(client_id2topk)):
            for j in range(len(client_id2topk[0])):
                for k in range(len(client_id2topk[0][0])):
                    orders[j][client_id2topk[i][j][k]] += 1
        for k in range(num_clients):
            orders[k] = dict(sorted(orders[k].items(), key=lambda item: item[1], reverse=True))

    for k1 in orders:
        print('{}: '.format(k1), end='')
        i = 0
        for k2, v2 in orders[k1].items():
            i += 1
            print('{}({})'.format(k2, v2), end=', ' if i < len(orders[k1]) else '\n')


if __name__ == '__main__':
    main()
