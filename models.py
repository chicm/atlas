import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from net.senet import se_resnext50_32x4d, se_resnet50, senet154, se_resnet152, se_resnext101_32x4d
from net.densenet import densenet121, densenet161, densenet169, densenet201
from net.nasnet import nasnetalarge
import settings


class ProteinNet(nn.Module):
    def __init__(self, backbone_name, num_classes=28, pretrained=True):
        super(ProteinNet, self).__init__()
        print('num_classes:', num_classes)
        if backbone_name in ['se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnet50', 'senet154', 'se_resnet152', 'nasnetmobile', 'mobilenet', 'nasnetalarge']:
            self.backbone = eval(backbone_name)()
        elif backbone_name in ['resnet34', 'resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201']:
            self.backbone = eval(backbone_name)(pretrained=pretrained)
            w = self.backbone.conv1.weight
            self.backbone.conv1 = nn.Conv2d(4, 64,kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, 2, :, :].unsqueeze(1)), dim=1))
        else:
            raise ValueError('unsupported backbone name {}'.format(backbone_name))
        #self.backbone.last_linear = nn.Linear(2048, 7272) # for model convert

        if backbone_name in ['resnet18', 'resnet34']:
            ftr_num = 512
        elif backbone_name =='nasnetmobile':
            ftr_num = 1056
        elif backbone_name == 'mobilenet':
            ftr_num = 1280
        elif backbone_name == 'densenet161':
            ftr_num = 2208
        elif backbone_name == 'densenet121':
            ftr_num = 1024
        elif backbone_name == 'densenet169':
            ftr_num = 1664
        elif backbone_name == 'densenet201':
            ftr_num = 1920
        elif backbone_name == 'nasnetalarge':
            ftr_num = 4032
        else:
            ftr_num = 2048

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(ftr_num, num_classes)
        self.name = 'ProteinNet_' + backbone_name
    
    def logits(self, x):
        x = self.avg_pool(x)
        x = F.dropout2d(x, p=0.4)
        x = x.view(x.size(0), -1)
        return self.logit(x)
    
    def forward(self, x):
        x = self.backbone.features(x)
        return self.logits(x)

    def get_logit_params(self, lr):
        group1 = [self.logit]

        params1 = []
        for x in group1:
            for p in x.parameters():
                params1.append(p)
        
        param_group1 = {'params': params1, 'lr': lr}

        return [param_group1]


def create_model(backbone):
    model = ProteinNet(backbone_name=backbone)
    model_file = os.path.join(settings.MODEL_DIR, model.name, 'best.pth')

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print('model file: {}, exist: {}'.format(model_file, os.path.exists(model_file)))

    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    
    return model, model_file

def test():
    model, _ = create_model('resnet34')
    x = torch.randn(4, 4, 512, 512).cuda()
    y = model(x)
    print(y.size())

if __name__ == '__main__':
    test()
    #convert_model4()
