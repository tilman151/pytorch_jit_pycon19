import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('torchvision_timings.pkl', mode='rb') as f:
    time_dict = pickle.load(f)

untraced_means = [np.mean(res['untraced']) for res in time_dict.values()]
untraced_stds = [np.std(res['untraced']) for res in time_dict.values()]
traced_means = [np.mean(res['traced']) for res in time_dict.values()]
traced_stds = [np.std(res['traced']) for res in time_dict.values()]
stats = [res['stats'] for res in time_dict.values()]

num_models = len(time_dict)
pos1 = np.arange(num_models // 2, 0, -1)
pos2 = pos1 - 0.4
pos_label = pos1 - 0.2
xlim = (max(*untraced_means, *traced_means) + max(*untraced_stds, *traced_stds)) * 1.05

plt.figure(figsize=(15, 7))
plt.style.use('ggplot')

for sub in range(1, 3):
    start = (sub - 1) * num_models // 2
    end = start + num_models // 2
    labels = list(time_dict.keys())[start:end]
    labels = [label + '\np=%.5f' % stat.pvalue for label, stat in zip(labels, stats[start:end])]

    plt.subplot(1, 2, sub)
    plt.ylim(0, num_models // 2 + 1)
    plt.yticks(pos_label, labels)
    plt.xlabel('ms')
    plt.xlim(right=xlim)
    plt.gcf().subplots_adjust(top=0.99, bottom=0.07, left=0.08, right=0.99, wspace=0.3)

    plt.barh(pos1, untraced_means[start:end], xerr=untraced_stds[start:end], capsize=3, height=0.4, label='untraced')
    plt.barh(pos2, traced_means[start:end], xerr=traced_stds[start:end], capsize=3, height=0.4, label='traced')
    plt.legend()

plt.savefig('torchvision_plot.png')
