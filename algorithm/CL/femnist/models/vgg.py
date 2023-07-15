import torch
import torch.nn as nn
import torch.nn.functional as F
import collections


cfg = {
    9: [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

BIAS_FLAG = False


class VGG_cifar(nn.Module):
    def __init__(self, depth, num_classes):
        super(VGG_cifar, self).__init__()
        self.layers = self._make_layers(cfg[depth])
        self.classifier = nn.Sequential(
            # nn.Linear(270848, num_classes, bias=BIAS_FLAG),
            nn.Linear(270848, num_classes),
        )
        self._initialize_weights()
        # self.regime = [
        #     {'epoch': 0, 'optimizer': 'SGD', 'lr': 0.01,
        #      'momentum': 0.9, 'weight_decay': 5e-4},
        #     # {'epoch': 200, 'lr': 0.001}
        #     {'epoch': 1000, 'optimizer': 'SGD', 'lr': 0.001,
        #      'momentum': 0.9, 'weight_decay': 5e-4},
        # ]
        # self.regime = [
        #     {'epoch': 0, 'optimizer': 'SGD', 'lr': 0.01,
        #      'momentum': 0.9, 'weight_decay': 5e-4},
        #     # {'epoch': 200, 'lr': 0.001}
        #     {'epoch': 10000, 'optimizer': 'SGD', 'lr': 0.001,
        #      'momentum': 0.9, 'weight_decay': 5e-4},
        # ]
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 0.01,
             'momentum': 0.9, 'weight_decay': 5e-4},
        ]



        # self.regime = [
        #     {'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1,
        #      'momentum': 0.9,'weight_decay': 5e-4 },
        #     {'epoch': 200, 'optimizer': 'SGD', 'lr': 0.02,
        #      'momentum': 0.9,'weight_decay': 5e-4 },
        #     {'epoch': 800, 'optimizer': 'SGD', 'lr': 0.004,
        #      'momentum': 0.9,'weight_decay': 5e-4 },
        #     {'epoch': 1000, 'optimizer': 'SGD', 'lr': 0.0008,
        #      'momentum': 0.9,'weight_decay': 5e-4 }
        #     ]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = [nn.Conv2d(1, 64, kernel_size=(3,3), padding=(1,1)),
                           #nn.BatchNorm2d(64),
                           nn.ReLU(inplace=True)]
        in_channels = 64
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           #nn.BatchNorm2d(x),
                           # nn.GroupNorm(1,x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


    def output_grad(self, opt):
        state_dict = collections.OrderedDict()
        opt_par = opt.optimizer.param_groups[0]["params"]
        for idx, l in enumerate(opt_par):
            if hasattr(l, 'grad'):
                layer_prefix = 'layers.' + str(idx) + '.'
                state_dict[layer_prefix + 'grad'] = l.grad
            else:
                assert (1==0)
                # state_dict[layer_prefix+'weight_exp']=l.weight_exp
            # if hasattr(l,'bias'):
            #     state_dict[layer_prefix+'bias']=l.bias
            #     state_dict[layer_prefix+'bias_exp']=l.bias_exp
        return state_dict


class VGG_imagenet(nn.Module):
    def __init__(self, depth):
        super(VGG_imagenet, self).__init__()
        self.layers = self._make_layers(cfg[depth])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1000),
        )
        self._initialize_weights()
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-2,
             'momentum': 0.9, 'weight_decay': 5e-4},
            # {'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
            {'epoch': 30, 'lr': 1e-3},
            {'epoch': 60, 'lr': 1e-4}]

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


class VGG_celeba(nn.Module):
    def __init__(self, num_features, num_classes):
        super(VGG_celeba, self).__init__()
        self.layers = self._make_layers(cfg[16])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096, bias=BIAS_FLAG),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(512*2*2, num_classes, bias=BIAS_FLAG),
        #     # nn.ReLU(),
        #     # nn.Linear(4096, 4096),
        #     # nn.ReLU(),
        #     # nn.Linear(4096, num_classes)
        # )
        self._initialize_weights()
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 0.001,
             'momentum': 0.9, 'weight_decay': 5e-4
             }
            # {'epoch': 100, 'lr': 0.001}
        ]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        # out = F.softmax(out, dim=1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=BIAS_FLAG),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
