import torch
import numpy as np


def MMD(x1, y1, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # x1, y1 = x.cpu().numpy(), y.cpu().numpy()
    xx1, yy1, zz1 = np.matmul(x1, np.transpose(x1)), np.matmul(y1, np.transpose(y1)), np.matmul(x1, np.transpose(y1))

    # xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (np.repeat(np.expand_dims(np.diag(xx1), axis=0), len(xx1), axis=0))
    ry = (np.repeat(np.expand_dims(np.diag(yy1), axis=0), len(yy1), axis=0))
    # ry = (np.diag(yy1).unsqueeze(0).expand_as(yy1))

    dxx = np.transpose(rx) + rx - 2. * xx1  # Used for A in (1)
    dyy = np.transpose(ry) + ry - 2. * yy1  # Used for B in (1)
    dxy = np.transpose(rx) + ry - 2. * zz1  # Used for C in (1)

    XX, YY, XY = (np.zeros(xx1.shape),
                  np.zeros(xx1.shape),
                  np.zeros(xx1.shape))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return np.mean(XX + YY - 2. * XY)