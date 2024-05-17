from thop import profile
import torch
from geoseg.models.CMTFNet import CMTFNet
from geoseg.models.ABCNet  import ABCNet
from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.networks.vision_mamba import MambaUnet as VIM

from config import get_config

from types import SimpleNamespace
args = {
    'root_path': '../data/ACDC',
    'exp': 'ACDC/Fully_Supervised',
    'model': 'VIM',
    'num_classes': 4,
    'cfg': '/root/folder/GeoSeg/config/vaihingen/vmamba_tiny.yaml',
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



def net_params(net):
    x = torch.randn(1, 3, 512, 512).cuda()
    net.eval()
    out = net(x)
    flops, params = profile(net, (x,))
    flops_gflops = flops / 1e9  # 从FLOPS换算到GigaFLOPS
    params_millions = params / 1e6  # 从参数数量换算到百万
    print(f"FLOPS: {flops_gflops} GFLOPS")
    print(f"Params: {params_millions} million")

# model = UNetFormer().eval().cuda()
# net_params(model)

config = get_config(args)
model = VIM(config, img_size=args.patch_size,
                    num_classes=6).cuda()


def print_module_parameters_in_millions(model):
    for name, module in model.named_children():
            for n, m in module.named_children(): 
                if n == "backbone" :
                    total_params = sum(p.numel() for p in m.parameters())
                    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
                    print(f"{n}: Total Parameters: {total_params / 1_000_000:.2f}M, Trainable Parameters: {trainable_params / 1_000_000:.2f}M")                          
                if n == "layers_up":
                    for n2, m2 in m.named_children():      
                        total_params = sum(p.numel() for p in m2.parameters())
                        trainable_params = sum(p.numel() for p in m2.parameters() if p.requires_grad)
                        print(f"{n2}: Total Parameters: {total_params / 1_000_000:.2f}M, Trainable Parameters: {trainable_params / 1_000_000:.2f}M")

print_module_parameters_in_millions(model)

net_params(model)
# print(model.mamba_unet.flops(shape=(3, 512, 512)))