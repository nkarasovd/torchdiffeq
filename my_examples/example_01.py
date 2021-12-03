import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Replace path to your own
sys.path.append('/Users/nikolajkarasov/Documents/neural_ode')

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=5000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., -0.1928, 7.6855]]).to(device)
# true_y0 = torch.tensor([[0., 0., 0.]]).to(device)
t = torch.linspace(0., 55., args.data_size).to(device)


class Lambda(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_pd = 1
        self.k_vco = 1
        self.tau_p = 0.9
        self.tau_z1 = 0.4
        self.tau_z2 = 0.4
        self.omega = 2
        self.k_f = 1.01

        self.coeff = (self.tau_z1 - self.tau_p) * (self.tau_p - self.tau_z2) / (self.tau_p ** 2)
        self.coeff_2 = self.tau_z1 * self.tau_z2 / self.tau_p * self.k_pd

        self.b = 1

    def misha_3d(self, t, y):
        s = self.k_pd * math.sin(y[:, 2])
        v = -1.0 / self.tau_p * y[:, 1] + self.coeff * self.k_pd * math.sin(y[:, 2])
        d = self.omega - self.k_f * self.k_vco * (y[:, 0] + y[:, 1] + self.coeff_2 * math.sin(y[:, 2]))

        return torch.tensor(np.array([s, v, d]), dtype=torch.float).reshape(1, 3)

    def forward(self, t, y):
        return self.misha_3d(t, y)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch():
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_phase = fig.gca(projection='3d')

    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')

        ax_phase.plot(true_y.cpu().numpy()[:, 0, 2], true_y.cpu().numpy()[:, 0, 1], true_y.cpu().numpy()[:, 0, 0], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 2], pred_y.cpu().numpy()[:, 0, 1], pred_y.cpu().numpy()[:, 0, 0],
                      'b--')
        ax_phase.set_xlim(-10, 10)
        ax_phase.set_ylim(-10, 10)

        fig.tight_layout()
        # plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.k_pd = 1
        self.k_vco = 1
        self.tau_p = 0.9
        self.tau_z1 = 0.4
        self.tau_z2 = 0.4
        self.omega = 2
        self.k_f = 1.01

        self.coeff = (self.tau_z1 - self.tau_p) * (self.tau_p - self.tau_z2) / (self.tau_p ** 2)
        self.coeff_2 = self.tau_z1 * self.tau_z2 / self.tau_p * self.k_pd

        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        if len(y.shape) == 2:
            s = self.k_pd * torch.sin(y[:, 2])
            v = -1.0 / self.tau_p * y[:, 1] + self.coeff * self.k_pd * torch.sin(y[:, 2])
            d = self.omega - self.k_f * self.k_vco * (y[:, 0] + y[:, 1] + self.coeff_2 * torch.sin(y[:, 2]))

            s = s.reshape(-1, 1)
            v = v.reshape(-1, 1)
            d = d.reshape(-1, 1)

            a = torch.cat((s, v, d), dim=1)

            return self.net(a)

        s = self.k_pd * torch.sin(y[:, 0, 2])
        v = -1.0 / self.tau_p * y[:, 0, 1] + self.coeff * self.k_pd * torch.sin(y[:, 0, 2])
        d = self.omega - self.k_f * self.k_vco * (y[:, 0, 0] + y[:, 0, 1] + self.coeff_2 * torch.sin(y[:, 0, 2]))

        s = s.reshape(-1, 1)
        v = v.reshape(-1, 1)
        d = d.reshape(-1, 1)

        a = torch.cat((s, v, d), dim=1).reshape(-1, 1, 3)

        return self.net(a)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)

    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()
