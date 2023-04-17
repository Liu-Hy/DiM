import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self,
                 num_classes,
                 net_norm='instance',
                 net_depth=3,
                 net_width=128,
                 channel=3,
                 net_act='relu',
                 net_pooling='avgpooling',
                 im_size=(32, 32)):
        # print(f"Define Convnet (depth {net_depth}, width {net_width}, norm {net_norm})")
        super(ConvNet, self).__init__()
        if net_act == 'sigmoid':
            self.net_act = nn.Sigmoid()
        elif net_act == 'relu':
            self.net_act = nn.ReLU()
        elif net_act == 'leakyrelu':
            self.net_act = nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

        if net_pooling == 'maxpooling':
            self.net_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            self.net_pooling = None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

        self.depth = net_depth
        self.net_norm = net_norm

        self.layers, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm,
                                                    net_pooling, im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, gt=None, return_features=False, debias_mode='none'):
        for d in range(self.depth):
            x = self.layers['conv'][d](x)
            if len(self.layers['norm']) > 0:
                x = self.layers['norm'][d](x)
            x = self.layers['act'][d](x)
            if len(self.layers['pool']) > 0:
                x = self.layers['pool'][d](x)

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
            output = self.classifier(x_new_view)
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
            x_new_view_after = self.classifier(x_new_view_after)
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
        # x = nn.functional.avg_pool2d(x, x.shape[-1])
        out = x.view(x.shape[0], -1)
        logit = self.classifier(out)

        if return_features:
            return logit, out
        else:
            return logit

    def get_feature(self, x, idx_from, idx_to=-1, return_prob=False, return_logit=False):
        if idx_to == -1:
            idx_to = idx_from
        features = []

        for d in range(self.depth):
            x = self.layers['conv'][d](x)
            if self.net_norm:
                x = self.layers['norm'][d](x)
            x = self.layers['act'][d](x)
            if self.net_pooling:
                x = self.layers['pool'][d](x)
            features.append(x)
            if idx_to < len(features):
                return features[idx_from:idx_to + 1]

        if return_prob:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            prob = torch.softmax(logit, dim=-1)
            return features, prob
        elif return_logit:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            return features, logit
        else:
            return features[idx_from:idx_to + 1]

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c * h * w)
        if net_norm == 'batch':
            norm = nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layer':
            norm = nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instance':
            norm = nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'group':
            norm = nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            norm = None
        else:
            norm = None
            exit('unknown net_norm: %s' % net_norm)
        return norm

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_pooling, im_size):
        layers = {'conv': [], 'norm': [], 'act': [], 'pool': []}

        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]

        for d in range(net_depth):
            layers['conv'] += [
                nn.Conv2d(in_channels,
                          net_width,
                          kernel_size=3,
                          padding=3 if channel == 1 and d == 0 else 1)
            ]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers['norm'] += [self._get_normlayer(net_norm, shape_feat)]
            layers['act'] += [self.net_act]
            in_channels = net_width
            if net_pooling != 'none':
                layers['pool'] += [self.net_pooling]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        layers['conv'] = nn.ModuleList(layers['conv'])
        layers['norm'] = nn.ModuleList(layers['norm'])
        layers['act'] = nn.ModuleList(layers['act'])
        layers['pool'] = nn.ModuleList(layers['pool'])
        layers = nn.ModuleDict(layers)

        return layers, shape_feat