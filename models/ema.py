import copy
import torch.nn as nn

class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        # 清理 GPU 缓存以释放内存
        import torch
        torch.cuda.empty_cache()
        
        # 方法1：尝试使用 state_dict 方式（更节省内存）
        # 但需要模型的构造函数，这可能不可行
        # 方法2：使用 deepcopy，但先清理缓存
        # 注意：deepcopy 是必要的，因为需要完全独立的模型实例
        
        # 先保存原始参数（用于恢复）
        original_params = {name: param.data.clone() for name, param in module.named_parameters()}
        
        # 创建模型副本
        try:
            # 尝试使用更节省内存的方式
            module_copy = copy.deepcopy(module)
        except Exception as e:
            # 如果 deepcopy 失败（内存不足），尝试清理更多缓存
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            # 再次尝试
            module_copy = copy.deepcopy(module)
        
        # 将 EMA 参数复制到新模型
        self.ema(module_copy)
        
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

