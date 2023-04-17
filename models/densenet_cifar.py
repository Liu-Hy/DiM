'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, gt=None, debias_mode='none'):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        x = F.avg_pool2d(F.relu(self.bn(out)), 4)

        num_rois = x.shape[0]
        num_channel = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]  # H * H
        HW = H * W

        if debias_mode in ["w2d", "rsc"]:
            drop_f = 1 / 3.0
            drop_b = 1 / 3.0
            self.eval()
            x_new = x.detach().clone().requires_grad_()
            x_new_view = x_new.view(x_new.size(0), -1)
            # print(x_new_view.shape)
            output = self.linear(x_new_view)
            class_num = output.shape[1]
            index = gt
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v,
                                                      torch.Size([num_rois, class_num])).to_dense().requires_grad_(
                requires_grad=False).cuda()
            # one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
            one_hot = torch.sum(output * one_hot_sparse)
            self.zero_grad()
            one_hot.backward()
            grads_val = x_new.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            spatial_mean = torch.sum(x_new * grad_channel_mean, 1)
            spatial_mean = spatial_mean.view(num_rois, HW)
            self.zero_grad()

            spatial_drop_num = int(HW * drop_f)
            th_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
            th_mask_value = th_mask_value.view(num_rois, 1).expand(num_rois, HW)
            mask_all_cuda = torch.where(spatial_mean > th_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                        torch.ones(spatial_mean.shape).cuda())
            mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)

            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            # x_new_view_after = self.avgpool(x_new_view_after)
            # x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            # x_new_view_after = self.fc(x_new_view_after)
            x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            x_new_view_after = self.linear(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
            th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][
                int(round(float(num_rois) * drop_b))]
            drop_index_fg = change_vector.gt(th_fg_value).long()
            ignore_index_fg = 1 - drop_index_fg
            not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
            mask_all[not_01_ignore_index_fg.long(), :] = 1
            self.train()
            # mask_all = Variable(mask_all, requires_grad=True)
            mask_all.requires_grad_()
            x = x * mask_all
        out = x.view(x.size(0), -1)
        out = self.linear(out)
        return out


def DenseNet121(nclass):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=nclass)


def DenseNet169(nclass):
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, num_classes=nclass)


def DenseNet201(nclass):
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32, num_classes=nclass)


def DenseNet161(nclass):
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48, num_classes=nclass)


def densenet_cifar(nclass):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=nclass)


def test():
    net = densenet_cifar()
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y)
