import random
import torch
import numpy as np
import bitsandbytes as bnb
from datetime import datetime
from transformers import set_seed


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def seed_everything(seed: int = 42) -> None:
    if seed:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_cur_time() -> str:
    """
    return: 1970-01-01
    """
    return datetime.now().strftime("%Y-%m-%d")


def get_cur_time_sec() -> str:
    """
    return: 1970-01-01 00:00:00
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
