import os
import os.path as osp
import re
import numpy as np
import json


num_seeds = 3
num_tasks = 3
log_dir = 'exp/sts_imdb_squad/'
methods = ['isolated', 'fedavg', 'fedavg_ft', 'fedprox',
           'fedbn', 'fedbn_ft', 'ditto', 'maml']
# num_rounds = [1, 50, 50, 50, 50, 50, 50, 50]

for m in methods:
    losses = []
    accs = []
    for i in range(1, num_seeds + 1):
        paths = [osp.join(log_dir, m, 'final', str(i), 'exp_print.log')]
        if m == 'maml':
            paths.append(osp.join(log_dir, m, 'final_ft', str(i), 'exp_print.log'))

        task_idx = 0
        tmp_losses = [[] for _ in range(num_tasks)]
        tmp_accs = [[] for _ in range(num_tasks)]
        for path in paths:
            update = False
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if re.search('Loss: \d+\.\d+', line) is None:
                        update = True
                        continue
                    if update:
                        task_idx += 1
                        update = False
                    task_i = (task_idx - 1) % num_tasks

                    cur_loss = re.search(r'Loss: (\d+\.\d+)', line)
                    cur_loss = float(cur_loss.groups()[0])
                    cur_acc = re.search(r'(Corr|Acc): (-?\d+\.\d+)', line)
                    cur_acc = float(cur_acc.groups()[-1])

                    tmp_losses[task_i].append(cur_loss)
                    tmp_accs[task_i].append(cur_acc)

        losses.append(tmp_losses)
        accs.append(tmp_accs)

    avg_loss = [np.mean(np.array(losses, dtype=object)[:, i].tolist(), axis=0).tolist() for i in range(len(losses))]
    avg_acc = [np.mean(np.array(accs, dtype=object)[:, i].tolist(), axis=0).tolist() for i in range(len(accs))]
    res = {'loss': avg_loss,
           'acc': avg_acc}

    # outdir = './training_curves'
    # os.makedirs(outdir, exist_ok=True)
    # with open(osp.join(outdir, m + '.json'), 'w') as f:
    #     json.dump(res, f, indent=2)

    pass
