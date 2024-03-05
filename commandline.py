import torch

from environments import get_envs


def get_device(args):
    device = args.device

    if device == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def add_default_args(parser):
    parser.add_argument('env', type=str, choices=get_envs())
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto')
