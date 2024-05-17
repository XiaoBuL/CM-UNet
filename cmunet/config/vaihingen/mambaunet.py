# import argparse

from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.vaihingen_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.networks.vision_mamba import MambaUnet as VIM
from catalyst.contrib.nn import Lookahead
from catalyst import utils

from config import get_config

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
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-4
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "ret-v3-lr-prehead-msa"
weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = weights_name # "mambanet-res-b4-512-crop-ms-e105-bs-4-gpu-4"

log_name = 'vaihingen/{}'.format(weights_name)
# monitor = 'val_F1'
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

# net.load_from(config)

# define the loss
weight = 0.4
loss = UnetFormerLoss(ignore_index=ignore_index, weight=weight)
use_aux_loss = True

# define the dataloader

train_dataset = VaihingenDataset(data_root='data/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(transform=val_aug)
test_dataset = VaihingenDataset(data_root='data/vaihingen/test',
                                transform=val_aug)

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
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

# for i, param_group in enumerate(base_optimizer.param_groups):
#     print(f"Parameter Group {i}:")
#     print(f"Learning Rate: {param_group['lr']}")
#     print(f"Weight Decay: {param_group['weight_decay']}")
# lr_scheduler = WarmupCosineAnnealingLR(optimizer, 15, max_epoch)

