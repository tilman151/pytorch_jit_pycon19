import torch
import torch.jit as jit
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
import pickle

from scipy.stats import f_oneway


DEVICE = 'cpu'
# DEVICE = 'cuda:0'


@torch.autograd.no_grad()
def profile_net(_net, _device):
    torch.cuda.empty_cache()
    _x = torch.randn(1, 3, 224, 224).to(_device)
    _runs = []
    _net(_x)
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            _net(_x)
        end.record()
        end.synchronize()
        if 'cuda' in _device:
            _runs.append(start.elapsed_time(end))
        else:
            _runs.append(start.elapsed_time(end))

    return _runs


def build_net(_net_func, _device):
    qualified_net_func = '.'.join(['torchvision', 'models', _net_func])
    qualified_net_func = eval(qualified_net_func)
    if not callable(qualified_net_func):
        return None
    _net = qualified_net_func().to(_device)
    _net.eval()

    return _net


if __name__ == '__main__':
    time_dict = {}
    networks = [func for func in dir(torchvision.models)
                if func.islower() and not func.startswith('_')]

    cudnn.benchmark = False
    cudnn.deterministic = True

    for net_func in networks:
        net = build_net(net_func, DEVICE)
        if net is None:
            continue
        print('Test %s' % net_func)

        runs = profile_net(net, DEVICE)
        print(np.mean(runs), np.std(runs))

        time_dict[net_func] = {'untraced': runs}

    for net_func in networks:
        net = build_net(net_func, DEVICE)
        if net is None:
            continue
        print('Test traced %s' % net_func)

        x = torch.randn(1, 3, 224, 224).to(DEVICE)
        traced_net = jit.trace(net, x)

        runs = profile_net(net, DEVICE)
        print(np.mean(runs), np.std(runs))

        time_dict[net_func]['traced'] = runs
        time_dict[net_func]['stats'] = f_oneway(time_dict[net_func]['untraced'], time_dict[net_func]['traced'])
        print(time_dict[net_func]['stats'])

    with open('torchvision_timings.pkl', mode='wb') as f:
        pickle.dump(time_dict, f)
