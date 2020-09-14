import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class LogScheduler(_LRScheduler):
    def __init__(self, optimizer, start_lr=0.03, end_lr=5e-4,
                 epochs=50, last_epoch=-1, **kwargs):
        self.lr_spaces = np.logspace(np.log10(start_lr),
                                     np.log10(end_lr),
                                     epochs)

        super(LogScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Mainly rewrite this method, to get lr for
        `scheduler.step()` usage
        Returns:
            current epoch lr, must iterable
        """
        # Because LogScheduler() initialization will call once step()
        if self.last_epoch == 0:
            epoch = self.last_epoch
        else:
            epoch = self.last_epoch - 1
        return [self.lr_spaces[epoch]]


if __name__ ==  '__main__':
    import torch.nn as nn
    from torch.optim import SGD
    import matplotlib.pyplot as plt

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 1, kernel_size=1)

    net = Net()
    # for i in net:
    #     print(i)
    optimizer = SGD(net.parameters(), lr=0.01)
    scheduler = LogScheduler(optimizer)
    lr = []
    for i in range(1, 51):
        lr.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    print(len(lr))
    plt.plot(lr)
    plt.show()