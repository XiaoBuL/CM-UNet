# import argparse

from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.loveda_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.networks.vision_mamba import MambaUnet as VIM
from config import get_config
from catalyst.contrib.nn import Lookahead
from catalyst import utils

from types import SimpleNamespace
args = {
    'root_path': '../data/ACDC',
    'exp': 'ACDC/Fully_Supervised',
    'model': 'VIM',
    'num_classes': 4,
    'cfg': '/data0/mushui/RemoteMamba/GeoSeg/config/vaihingen/vmamba_tiny.yaml',
    'opts': None,  # This is a list and will be None by default
    'zip': False,  # False by default, true if --zip is used
    'cache_mode': 'part',  # Default is 'part'
    'resume': None,  # No default provided, so it's set to None
    'accumulation_steps': None,  # No default provided, so it's set to None
    'use_checkpoint': False,  # False by default, true if --use-checkpoint is used
    'amp_opt_level': 'O1',
    'tag': None,  # No default provided, so it's set to None
    'eval': False,  # False by default, true if --eval is used
    'throughput': False,  # False by default, true if --throughput is used
    'max_iterations': 10000,
    'batch_size': 24,
    'deterministic': 1,
    'base_lr': 0.01,
    'patch_size': 4,
    'seed': 1337,
    'labeled_num': 140
}
args = SimpleNamespace(**args)


# training hparam
max_epoch = 105
ignore_index = len(CLASSES)
train_batch_size = 16 # todo
val_batch_size = 16
lr = 6e-4 # todo
weight_decay = 0.01 # todo
backbone_lr = 6e-5
backbone_weight_decay = 0.01 # todo
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "mambaunet-res-b-512-crop-ms-e45-8e-4-large"
weights_path = "model_weights/loveda/{}".format(weights_name)
test_weights_name = f"{weights_name}"

log_name = 'loveda/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
# net = UNetFormer(num_classes=num_classes)

config = get_config(args)
net = VIM(config, img_size=args.patch_size,
                    num_classes=num_classes)

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True


def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


train_dataset = LoveDATrainDataset(transform=train_aug, data_root='data/LoveDA/Train')

val_dataset = loveda_val_dataset

test_dataset = loveda_val_dataset

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)



class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            lr = [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # 余弦退火调整
            cos_epoch = self.last_epoch - self.warmup_epochs
            cos_epochs = self.max_epochs - self.warmup_epochs
            lr = [base_lr * (1 + math.cos(math.pi * cos_epoch / cos_epochs)) / 2 for base_lr in self.base_lrs]
        return lr
def print_model_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_size = parameter.numel()  # Number of elements in the parameter
            total_params += param_size
            # print(f"{name}: shape={parameter.size()} count={param_size}")
    total_params_million = total_params / 1_000_000  # Convert to millions
    print(f"Total trainable parameters: {total_params} ({total_params_million:.2f}M)")


def print_module_parameters_in_millions(model):
    for name, module in model.named_children():
            for n, m in module.named_children():        
                if n == "layers_up":
                    for n2, m2 in m.named_children():      
                        total_params = sum(p.numel() for p in m2.parameters())
                        trainable_params = sum(p.numel() for p in m2.parameters() if p.requires_grad)
                        print(f"{n2}: Total Parameters: {total_params / 1_000_000:.2f}M, Trainable Parameters: {trainable_params / 1_000_000:.2f}M")


# print_module_parameters_in_millions(net)
print_model_parameters(net)


# define the optimizer
layerwise_params = {"mamba_unet.backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

from thop import profile
x = torch.randn(1, 3, 256, 256).cuda()
net.cuda()
net.eval()
# print(net.mamba_unet.flops())
out = net(x)
flops, params = profile(net, (x,))
print(flops / 1e9)
print(params / 1e6)

import time
x = torch.zeros((1,3,256,256)).cuda()
t_all = []

for i in range(1000):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))

raise ValueError
