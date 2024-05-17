import torch
import numpy as np

from thop import profile
import time
from geoseg.models.ABCNet import ABCNet
from geoseg.models.CMTFNet import CMTFNet
from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.networks.vision_mamba import MambaUnet as VIM
from config import get_config
from geoseg.models.FTUNetFormer import ft_unetformer


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

config = get_config(args)



def calculate(net):
    x = torch.randn(2, 3, 256, 256).cuda()
    net.cuda()
    net.eval()
    # print(net.mamba_unet.flops())
    out = net(x)
    flops, params = profile(net, (x,))
    print(flops / 1e9)
    print(params / 1e6)
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # 转换为 MB
    print("Max memory allocated:", max_memory_allocated, "MB")
    # 重置最大内存计数
    torch.cuda.reset_max_memory_allocated()

    x = torch.zeros((1,3,256,256)).cuda()
    t_all = []
    for i in range(1000):
        t1 = time.time()
        y = net(x)
        t2 = time.time()
        t_all.append(t2 - t1)
    print('average fps:',1 / np.mean(t_all))

# print("=====ABCNet======")
# abc_net = ABCNet(n_classes=6)
# calculate(abc_net)

# print("=====CMTFNet======")
# cmtfnet = CMTFNet(num_classes=6)
# calculate(cmtfnet)

# print("=====UNetFormer======")
# unetformer = UNetFormer(num_classes=6)
# calculate(unetformer)

# print("=====ft_unetformer======")
# ftunetformer = ft_unetformer(num_classes=6, decoder_channels=256)
# calculate(ftunetformer)

print("=====VIM======")
mamba = VIM(config, img_size=4, num_classes=6)
calculate(mamba)


# print('fastest time:', min(t_all) / 1)
# print('fastest fps:',1 / min(t_all))

# print('slowest time:', max(t_all) / 1)
# print('slowest fps:',1 / max(t_all))
