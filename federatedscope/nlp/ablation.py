import os
import os.path as osp
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


def abl1():
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


def abl2():
    def _parse_str(str):
        groups = [s.split(': ') for s in ']\t'.join(str.split('], ')).split('\t')]
        groups = [(int(g[0]), [int(x) for x in g[1][1:-1].split(', ')]) for g in groups]
        g2c, c2g = {}, {}
        for g in groups:
            g2c[g[0]] = g[1]
            for c in g[1]:
                c2g[c] = g[0]
        return g2c, c2g

    n_client = 18
    exp_path = osp.join(osp.dirname(osp.abspath(os.getcwd())), 'exp/pfednlp_pt/v5/bert2bert/200_50/group_5/pretrain/exp_print.log')
    cnt = {k1: {k2: 0 for k2 in range(1, n_client + 1)} for k1 in range(1, n_client + 1)}
    with open(exp_path) as f:
        for line in f:
            res = re.search(r'group_id2client: (.+)', line)
            if res is not None:
                s = res.groups()[0][1:-1]
                g2c, c2g = _parse_str(s)
                n_client = len(c2g)
                for i in range(1, n_client + 1):
                    for j in range(1, n_client + 1):
                        if c2g[i] == c2g[j]:
                            cnt[i][j] += 1
    for k in range(1, n_client + 1):
        sorted_dict = dict(sorted(cnt[k].items(), key=lambda item: item[1], reverse=True))
        keys = list(sorted_dict.keys())
        idx = 0
        while keys[idx] != k:
            idx += 1
        if idx != 0:
            tmp = keys[0]
            keys[0] = k
            keys[idx] = tmp
        cnt[k] = {kk: sorted_dict[kk] for kk in keys}

    n_group = [1, 3, 3, 2, 5, 4]
    start_id = 1
    for gi, num in enumerate(n_group):
        cur_cnt = {k1: 0 for k1 in range(1, n_client + 1)}
        for i in range(start_id, start_id + num):
            for k in range(1, n_client + 1):
                cur_cnt[k] += cnt[i][k] / num
        cnt['g_{}'.format(gi + 1)] = dict(sorted(cur_cnt.items(), key=lambda item: item[1], reverse=True))
        start_id += num

    for k1 in cnt:
        print('{}: '.format(k1), end='')
        i = 0
        for k2, v2 in cnt[k1].items():
            i += 1
            print('{}({:.1f})'.format(k2, v2), end=', ' if i < len(cnt[k1]) else '\n')

    return cnt


def abl3():
    def _parse_str(str):
        groups = [s.split(': ') for s in ']\t'.join(str.split('], ')).split('\t')]
        groups = [(int(g[0]), [int(x) for x in g[1][1:-1].split(', ')]) for g in groups]
        c2topk = {}
        for g in groups:
            c2topk[g[0]] = g[1][:topk]
        return c2topk

    n_client = 18
    # topk = 6 #10
    exp_path = osp.join(osp.dirname(osp.abspath(os.getcwd())), 'exp/pfednlp_ft/v5/bert2bert/pt_g5_ft_t16/train/exp_print.log')
    cnt = {k1: {k2: 0 for k2 in range(1, n_client + 1)} for k1 in range(1, n_client + 1)}
    with open(exp_path) as f:
        for line in f:
            res = re.search(r'client_id2topk: (.+)', line)
            if res is not None:
                s = res.groups()[0][1:-1]
                c2topk = _parse_str(s)
                n_client = len(c2topk)
                for i in range(1, n_client + 1):
                    for j in range(1, n_client + 1):
                        if j in c2topk[i]:
                            cnt[i][j] += 1
    # for k in range(1, n_client + 1):
    #     cnt[k] = dict(sorted(cnt[k].items(), key=lambda item: item[1], reverse=True))
    for k in range(1, n_client + 1):
        sorted_dict = dict(sorted(cnt[k].items(), key=lambda item: item[1], reverse=True))
        keys = list(sorted_dict.keys())
        idx = 0
        while keys[idx] != k:
            idx += 1
        if idx != 0:
            tmp = keys[0]
            keys[0] = k
            keys[idx] = tmp
        cnt[k] = {kk: sorted_dict[kk] for kk in keys}

    n_group = [1, 3, 3, 2, 5, 4]
    start_id = 1
    for gi, num in enumerate(n_group):
        cur_cnt = {k1: 0 for k1 in range(1, n_client + 1)}
        for i in range(start_id, start_id + num):
            for k in range(1, n_client + 1):
                cur_cnt[k] += cnt[i][k] / num
        cnt['g_{}'.format(gi + 1)] = dict(sorted(cur_cnt.items(), key=lambda item: item[1], reverse=True))
        start_id += num

    for k1 in cnt:
        print('{}: '.format(k1), end='')
        i = 0
        for k2, v2 in cnt[k1].items():
            i += 1
            print('{}({:.1f})'.format(k2, v2), end=', ' if i < len(cnt[k1]) else '\n')

    return cnt


def abl4():
    def _parse_str(str):
        groups = [s.split(': ') for s in ']\t'.join(str.split('], ')).split('\t')]
        groups = [(int(g[0]), [int(x) for x in g[1][1:-1].split(', ')]) for g in groups]
        c2topk = {}
        for g in groups:
            c2topk[g[0]] = g[1][:topk]
        return c2topk

    n_client = 18
    # topk = 6
    # exp_path = osp.join(osp.dirname(osp.abspath(os.getcwd())), 'exp/pfednlp_ft/v5/bert2bert/contrast_dec/no_share_dec/grad/pt_g5_ft_t16_gen_t8/train/exp_print.log')
    exp_path = osp.join(osp.dirname(osp.abspath(os.getcwd())), 'exp/pfednlp_ft/v5/bert2bert/contrast_dec/no_share_dec/grad_proto_weight_grad/proto_w_0.1/pt_g5_ft_t16_gen_t8/train/exp_print.log')
    cnt = {k1: {k2: 0 for k2 in range(1, n_client + 1)} for k1 in range(1, n_client + 1)}
    with open(exp_path) as f:
        for line in f:
            res = re.search(r'client_id2all \(.+\): (.+)', line)
            # res = re.search(r'client_id2all: (.+)', line)
            if res is not None:
                s = res.groups()[0][1:-1]
                c2topk = _parse_str(s)
                n_client = len(c2topk)
                for i in range(1, n_client + 1):
                    for j in range(1, n_client + 1):
                        if j in c2topk[i]:
                            cnt[i][j] += 1
    for k in range(1, n_client + 1):
        cnt[k] = dict(sorted(cnt[k].items(), key=lambda item: item[1], reverse=True))

    n_group = [1, 3, 3, 2, 5, 4]
    start_id = 1
    for gi, num in enumerate(n_group):
        cur_cnt = {k1: 0 for k1 in range(1, n_client + 1)}
        for i in range(start_id, start_id + num):
            for k in range(1, n_client + 1):
                cur_cnt[k] += cnt[i][k] / num
        cnt['g_{}'.format(gi + 1)] = dict(sorted(cur_cnt.items(), key=lambda item: item[1], reverse=True))
        start_id += num

    for k1 in cnt:
        print('{}: '.format(k1), end='')
        i = 0
        for k2, v2 in cnt[k1].items():
            i += 1
            print('{}({:.1f})'.format(k2, v2), end=', ' if i < len(cnt[k1]) else '\n')

    return cnt


def plot(agg_type):
    if agg_type == 'coarse_agg':
        cnt = abl2()
    elif agg_type == 'fine_agg':
        cnt = abl3()
    elif agg_type == 'fine_agg_contrast':
        cnt = abl4()

    width = 0.04
    n_top = 10
    client_ids = [[8, 3], [15, 10]]
    clients = [{i: cnt[i] for i in ids} for ids in client_ids]
    # ind = np.arange(len(client_ids[0]))
    ind = [0, 0.45]
    colors = {1: 'tab:blue',
              2: 'tab:orange',
              3: 'tab:orange',
              4: 'tab:orange',
              5: 'tab:green',
              6: 'tab:green',
              7: 'tab:green',
              8: 'tab:red',
              9: 'tab:red',
              10: 'tab:purple',
              11: 'tab:purple',
              12: 'tab:purple',
              13: 'tab:purple',
              14: 'tab:purple',
              15: 'tab:cyan',
              16: 'tab:cyan',
              17: 'tab:cyan',
              18: 'tab:cyan'}
    hatches = {1: None,
               2: '//',
               3: '//',
               4: '//',
               5: '\\\\',
               6: '\\\\',
               7: '\\\\',
               8: '--',
               9: '--',
               10: '..',
               11: '..',
               12: '..',
               13: '..',
               14: '..',
               15: '**',
               16: '**',
               17: '**',
               18: '**'}

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.axisbelow'] = True
    fontsize = 12
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
    for fi in range(2):
        ind_ = ind
        for j in range(n_top):
            cur_ids = [list(clients[fi][i].keys())[n_top - j - 1] for i in client_ids[fi]]
            cur_cnts = [list(clients[fi][i].values())[n_top - j - 1] for i in client_ids[fi]]
            ax[fi].barh(ind_, cur_cnts, width,
                        color=[colors[k] for k in cur_ids],
                        hatch=[hatches[k] for k in cur_ids],
                        edgecolor='k')
            ind_ = [x + width for x in ind_]

        y_ind = [[x + j * width for x in ind] for j in range(n_top)]
        y_val = [[list(clients[fi][i].keys())[n_top - j - 1] for i in client_ids[fi]] for j in range(n_top)]
        y_ind = [x for subl in y_ind for x in subl]
        y_val = [x for subl in y_val for x in subl]
        ax[fi].set_yticks(y_ind, y_val, fontsize=0.8 * fontsize)
        ax[fi].tick_params(axis='x', labelsize=0.8 * fontsize)
        ax[fi].grid(axis='x')

    # plt.ylabel('Client ID', fontsize=fontsize)
    # plt.xlabel('Co-occurance Frequency', fontsize=fontsize)
    fig.text(0.45, 0, 'Co-occurance Frequency', ha='center', va='center', fontsize=fontsize)
    fig.text(0, 0.5, 'Client ID', ha='center', va='center', rotation='vertical', fontsize=fontsize)

    legend_ids = [1, 2, 5, 8, 10, 15]
    labels = ['IMDB', 'AGNews', 'SQuAD', 'NewsQA', 'CNN/DM', 'MSQG']
    handles = [mpatches.Patch(alpha=0.8, facecolor=colors[label], hatch=hatches[label], label=labels[i])
               for i, label in enumerate(legend_ids)]
    plt.legend(handles=handles, fontsize=0.8 * fontsize, loc='center left', bbox_to_anchor=(1, 0.8))
    plt.tight_layout()
    plt.savefig('./figs/{}.pdf'.format(agg_type), bbox_inches='tight')
    plt.show()


def plot2():
    cnt1 = abl3()  # fine-agg
    cnt2 = abl4()  # fine-agg-contrast

    n_top = 10
    client_ids = [10, 15]
    gen_ids = list(range(10, 19))
    ind = np.arange(1, n_top + 1)
    linestyles = ['--', '-']  # fine-agg & fine-agg-contrast
    markers = ['o', 'x']  # gen & non-gen
    colors = ['tab:blue', 'tab:orange']  # gen & non-gen
    line_color = 'tab:gray'
    marker_width = 2

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.axisbelow'] = True
    fontsize = 18
    text_offset = 0.3
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    for fi in range(2):
        ids1 = list(cnt1[client_ids[fi]].keys())[:n_top]
        freq1 = list(cnt1[client_ids[fi]].values())[:n_top]
        gen_ind1 = [x for x in range(1, n_top + 1) if ids1[x - 1] in gen_ids]
        gen_freq1 = [freq1[x - 1] for x in gen_ind1]
        non_gen_ind1 = [x for x in range(1, n_top + 1) if ids1[x - 1] not in gen_ids]
        non_gen_freq1 = [freq1[x - 1] for x in non_gen_ind1]
        ax[fi].plot(ind, freq1, linestyle=linestyles[0], color=line_color)
        for x, y in zip(gen_ind1, gen_freq1):
            ax[fi].plot(x, y, marker=markers[0], mfc=colors[0], mec=colors[0], mew=marker_width)
            ax[fi].annotate(str(ids1[x - 1]), xy=(x + text_offset, y + text_offset), fontsize=0.8 * fontsize)
        for x, y in zip(non_gen_ind1, non_gen_freq1):
            ax[fi].plot(x, y, marker=markers[1], mfc=colors[1], mec=colors[1], mew=marker_width)
            ax[fi].annotate(str(ids1[x - 1]), xy=(x + text_offset, y + text_offset), fontsize=0.8 * fontsize)

        ids2 = list(cnt2[client_ids[fi]].keys())[:n_top]
        freq2 = list(cnt2[client_ids[fi]].values())[:n_top]
        gen_ind2 = [x for x in range(1, n_top + 1) if ids2[x - 1] in gen_ids]
        gen_freq2 = [freq2[x - 1] for x in gen_ind2]
        non_gen_ind2 = [x for x in range(1, n_top + 1) if ids2[x - 1] not in gen_ids]
        non_gen_freq2 = [freq2[x - 1] for x in non_gen_ind2]
        ax[fi].plot(ind, freq2, linestyle=linestyles[1], color=line_color)
        for x, y in zip(gen_ind2, gen_freq2):
            ax[fi].plot(x, y, marker=markers[0], mfc=colors[0], mec=colors[0], mew=marker_width)
            ax[fi].annotate(str(ids2[x - 1]), xy=(x + text_offset, y + text_offset), fontsize=0.8 * fontsize)
        for x, y in zip(non_gen_ind2, non_gen_freq2):
            ax[fi].plot(x, y, marker=markers[1], mfc=colors[1], mec=colors[1], mew=marker_width)
            ax[fi].annotate(str(ids2[x - 1]), xy=(x + text_offset, y + text_offset), fontsize=0.8 * fontsize)

        ax[fi].grid()
        if fi == 1:
            ax[fi].set_yticklabels([])
        ax[fi].tick_params(labelsize=0.8 * fontsize)

    fig.text(0.45, 0, 'Ranking Index', ha='center', va='center', fontsize=fontsize)
    fig.text(0, 0.5, 'Co-occurance Frequency', ha='center', va='center', rotation='vertical', fontsize=fontsize)

    lines = [Line2D([0], [0], linestyle=linestyles[0], marker=markers[0], color=line_color, mfc=colors[0], mec=colors[0], mew=marker_width),
             Line2D([0], [0], linestyle=linestyles[0], marker=markers[1], color=line_color, mfc=colors[1], mec=colors[1], mew=marker_width),
             Line2D([0], [0], linestyle=linestyles[1], marker=markers[0], color=line_color, mfc=colors[0], mec=colors[0], mew=marker_width),
             Line2D([0], [0], linestyle=linestyles[1], marker=markers[1], color=line_color, mfc=colors[1], mec=colors[1], mew=marker_width)]
    labels = ['w/o PCCL\nNLG',
              'w/o PCCL\nNLU',
              'w/ PCCL\nNLG',
              'w/ PCCL\nNLU']
    plt.legend(lines, labels, fontsize=0.8 * fontsize, loc='center left', bbox_to_anchor=(1, 0.7))

    plt.tight_layout()
    plt.savefig('./figs/contrast_improve.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    topk = 8

    # agg_type = 'coarse_agg'
    # plot(agg_type)
    #
    # agg_type = 'fine_agg'
    # plot(agg_type)

    # agg_type = 'fine_agg_contrast'
    # plot(agg_type)

    plot2()
