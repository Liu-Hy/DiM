import torch
import random
import numpy as np
import os
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import DGdatasets
from torchvision import datasets, transforms


matplotlib.use('Agg')

__all__ = ["Compose", "Lighting", "ColorJitter"]

def load_data(args):
    dsts = None
    if args.data.lower() == 'mnist':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif args.data.lower() == 'cmnist':
        channel = 2
        im_size = (28, 28)
        num_classes = 2
        mean = [0.1307, 0.1307]
        std = [0.3081, 0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dsts = DGdatasets.ColoredMNIST(args.data_dir).datasets
        class_names = [str(c) for c in range(num_classes)]

    elif args.data.lower() == 'pacs':
        channel = 3
        im_size = (32, 32)
        num_classes = 7
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dsts = DGdatasets.PACS(args.data_dir).datasets
        class_names = dsts[1].classes

    elif args.data.lower() == 'vlcs':
        channel = 3
        im_size = (32, 32)
        num_classes = 5
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dsts = DGdatasets.VLCS(args.data_dir).datasets
        class_names = dsts[1].classes
        print(class_names)

    elif args.data.lower() == 'officehome':
        channel = 3
        im_size = (64, 64)
        num_classes = 65
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dsts = DGdatasets.OfficeHome(args.data_dir).datasets
        class_names = dsts[1].classes
        print(class_names)

    elif args.data.lower() == 'fashionmnist':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(args.data_dir, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(args.data_dir, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif args.data.lower() == 'svhn':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(args.data_dir, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(args.data_dir, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif args.data.lower() == 'cifar10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif args.data.lower() == 'cifar100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(args.data_dir, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(args.data_dir, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif args.data.lower() == 'tinyimagenet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        d_path = os.path.join(args.data_dir, "tiny-imagenet")
        dst_train = datasets.ImageFolder(os.path.join(d_path, "train"), transform=transform)
        dst_test = datasets.ImageFolder(os.path.join(d_path, "val"), transform=transform)
        class_names = dst_train.classes
    else:
        exit('unknown dataset: %s'%args.data)

    # testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    if dsts is None:
        dsts = dst_train, dst_test
    return channel, im_size, num_classes, class_names, mean, std, dsts



class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def dist_l2(data, target):
    dist = (data**2).sum(-1).unsqueeze(1) + (
        target**2).sum(-1).unsqueeze(0) - 2 * torch.matmul(data, target.transpose(1, 0))
    return dist


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class TimeStamp():
    def __init__(self, print_log=True):
        self.prev = time.time()
        self.print_log = print_log
        self.times = {}

    def set(self):
        self.prev = time.time()

    def flush(self):
        if self.print_log:
            print("\n=========Summary=========")
            for key in self.times.keys():
                times = np.array(self.times[key])
                print(
                    f"{key}: {times.sum():.4f}s (avg {times.mean():.4f}s, std {times.std():.4f}, count {len(times)})"
                )
                self.times[key] = []

    def stamp(self, name=''):
        if self.print_log:
            spent = time.time() - self.prev
            # print(f"{name}: {spent:.4f}s")
            if name in self.times.keys():
                self.times[name].append(spent)
            else:
                self.times[name] = [spent]
            self.set()


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Plotter():
    def __init__(self, path, nepoch, idx=0):
        self.path = path
        self.data = {'epoch': [], 'acc_tr': [], 'acc_val': [], 'loss_tr': [], 'loss_val': []}
        self.nepoch = nepoch
        self.plot_freq = 10
        self.idx = idx

    def update(self, epoch, acc_tr, acc_val, loss_tr, loss_val):
        self.data['epoch'].append(epoch)
        self.data['acc_tr'].append(acc_tr)
        self.data['acc_val'].append(acc_val)
        self.data['loss_tr'].append(loss_tr)
        self.data['loss_val'].append(loss_val)

        if len(self.data['epoch']) % self.plot_freq == 0:
            self.plot()

    def plot(self, color='black'):
        fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 3))
        fig.tight_layout(h_pad=3, w_pad=3)

        fig.suptitle(f"{self.path}", size=16, y=1.1)

        axes[0].plot(self.data['epoch'], self.data['acc_tr'], color, lw=0.8)
        axes[0].set_xlim([0, self.nepoch])
        axes[0].set_ylim([0, 100])
        axes[0].set_title('acc train')

        axes[1].plot(self.data['epoch'], self.data['acc_val'], color, lw=0.8)
        axes[1].set_xlim([0, self.nepoch])
        axes[1].set_ylim([0, 100])
        axes[1].set_title('acc val')

        axes[2].plot(self.data['epoch'], self.data['loss_tr'], color, lw=0.8)
        axes[2].set_xlim([0, self.nepoch])
        axes[2].set_ylim([0, 3])
        axes[2].set_title('loss train')

        axes[3].plot(self.data['epoch'], self.data['loss_val'], color, lw=0.8)
        axes[3].set_xlim([0, self.nepoch])
        axes[3].set_ylim([0, 3])
        axes[3].set_title('loss val')

        for ax in axes:
            ax.set_xlabel('epochs')

        plt.savefig(f'{self.path}/curve_{self.idx}.png', bbox_inches='tight')
        plt.close()


def random_indices(y, nclass=10, intraclass=False, device='cuda'):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n).to(device)
    return index


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""
    def __init__(self, alphastd, eigval, eigvec, device='cpu'):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval, device=device)
        self.eigvec = torch.tensor(eigvec, device=device)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        # make differentiable
        if len(img.shape) == 4:
            return img + rgb.view(1, 3, 1, 1).expand_as(img)
        else:
            return img + rgb.view(3, 1, 1).expand_as(img)


class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)


class CutOut():
    def __init__(self, ratio, device='cpu'):
        self.ratio = ratio
        self.device = device

    def __call__(self, x):
        n, _, h, w = x.shape
        cutout_size = [int(h * self.ratio + 0.5), int(w * self.ratio + 0.5)]
        offset_x = torch.randint(h + (1 - cutout_size[0] % 2), size=[1], device=self.device)[0]
        offset_y = torch.randint(w + (1 - cutout_size[1] % 2), size=[1], device=self.device)[0]

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(n, dtype=torch.long, device=self.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=self.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=self.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=h - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=w - 1)
        mask = torch.ones(n, h, w, dtype=x.dtype, device=self.device)
        mask[grid_batch, grid_x, grid_y] = 0

        x = x * mask.unsqueeze(1)
        return x


class Normalize():
    def __init__(self, mean, std, device='cpu'):
        self.mean = torch.tensor(mean, device=device).reshape(1, len(mean), 1, 1)
        self.std = torch.tensor(std, device=device).reshape(1, len(mean), 1, 1)

    def __call__(self, x, seed=-1):
        return (x - self.mean) / self.std
