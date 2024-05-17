from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.potsdam_dataset import *
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
max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-4
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

test_time_aug = 'd4'
output_mask_dir, output_mask_rgb_dir = None, None
weights_name = "e45"
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = f"{weights_name}"
log_name = 'potsdam/{}'.format(weights_name)
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
config = get_config(args)
net = VIM(config, img_size=args.patch_size,
                    num_classes=num_classes)

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader

train_dataset = PotsdamDataset(data_root='data/potsdam/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(transform=val_aug)
test_dataset = PotsdamDataset(data_root='data/potsdam/test',
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

def print_model_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_size = parameter.numel()  # Number of elements in the parameter
            total_params += param_size
            # print(f"{name}: shape={parameter.size()} count={param_size}")
    total_params_million = total_params / 1_000_000  # Convert to millions
    print(f"Total trainable parameters: {total_params} ({total_params_million:.2f}M)")


# print_module_parameters_in_millions(net)
print_model_parameters(net)

# define the optimizer
layerwise_params = {"mamba_unet.backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)