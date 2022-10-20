import numpy as np


def main():
    res = '77.72, 92.32, 41.76/46.60, 21.84/34.43, 37.21/15.28/33.51, 25.96/2.45/15.14'
    all_score = []
    for score in res.split(', '):
        if len(score.split('/')) == 1:
            cur_score = float(score)
        else:
            cur_score = np.mean([float(x) for x in score.split('/')])
        print(cur_score)
        all_score.append(cur_score)
    print('------')
    avg_score2 = [np.mean(all_score[i: i + 2]) for i in range(0, 6, 2)]
    for s in avg_score2:
        print(s)
    print('--- Avg ---')
    avg_score = np.mean(all_score)
    print(avg_score)


if __name__ == '__main__':
    main()
