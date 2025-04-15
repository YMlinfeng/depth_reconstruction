# training/schedulers.py
import math
from torch.optim.lr_scheduler import LambdaLR, StepLR

# =====================================================
#  封装类：封装 LambdaLR，提供统一接口
# =====================================================
class WarmupCosineLRScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr=1e-3, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / warmup_steps
            else:
                progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                return (self.min_lr + (self.base_lr - self.min_lr) * cosine) / self.base_lr

        self.scheduler = LambdaLR(optimizer, lr_lambda)

    def step(self):
        self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    

# =====================================================
#  统一封装类：封装 LambdaLR 或 StepLR，并提供统一接口
# =====================================================
class SchedulerWrapper:
    def __init__(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler  

    def step(self):
        self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

    def get_lr(self):
        # 返回当前第一个 param_group 的学习率（通常就是主 lr）
        return self.optimizer.param_groups[0]['lr']
    
    def get_lr_dict(self):
        # 返回所有 param_group 的学习率
        return {i: group['lr'] for i, group in enumerate(self.optimizer.param_groups)}
    
    def get_lr_scale_dict(self):
        # 返回所有 param_group 的学习率缩放倍率
        return {i: group['lr_scale'] for i, group in enumerate(self.optimizer.param_groups)}
    

# SchedulerFactory：根据策略名返回 SchedulerWrapper 实例
class SchedulerFactory:
    def __init__(self, optimizer, total_steps, warmup_steps=0, base_lr=1e-3, min_lr=2e-7):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr

    def get(self, name="cosine"):
        name = name.lower()

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return current_step / float(max(1, self.warmup_steps))
            
            progress = (current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            progress = min(progress, 1.0)

            if name == "cosine":
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
                return lr / self.base_lr

            elif name == "linear":
                lr = self.min_lr + (self.base_lr - self.min_lr) * (1 - progress)
                return lr / self.base_lr

            elif name == "constant":
                return 1.0

            else:
                raise ValueError(f"Unsupported scheduler: {name}")

        # StepLR 特殊处理（不支持 warmup）
        if name == "step":
            scheduler = StepLR(self.optimizer, step_size=int(self.total_steps * 0.3), gamma=0.1)
        else:
            scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        #  返回统一封装类
        return SchedulerWrapper(self.optimizer, scheduler)