import torch
from torch import nn
from torch.nn.utils.spectral_norm import SpectralNorm
from copy import deepcopy

from networks import utils as nt_utils



############################################################
#                 Utils for options parsing                #
############################################################

def str2bool(string):
    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        raise

def parse_str_to_list(string, value_type=str, sep=','):
    if string:
        outputs = string.replace(' ', '').split(sep)
    else:
        outputs = []
    outputs = [value_type(output) for output in outputs]
    return outputs

def parse_str_to_dict(string, value_type=str, sep=','):
    items = [s.split(':') for s in string.replace(' ', '').split(sep)]
    return {k: value_type(v) for k, v in items}

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def set_batchnorm_momentum(module, momentum):
    from apex import parallel

    if isinstance(module, (nn.BatchNorm2d, parallel.SyncBatchNorm)):
        module.momentum = momentum

def parse_args_line(line):
    # Parse a value from string
    parts = line[:-1].split(': ')
    if len(parts) > 2:
        parts = [parts[0], ': '.join(parts[1:])]
    k, v = parts
    v_type = str
    if v.isdigit():
        v = int(v)
        v_type = int
    elif isfloat(v) and k != 'per_loss_weights' and k != 'pix_loss_weights': # per_loss_weights can be a string of floats
        v_type = float
        v = float(v)
    elif v == 'True':
        v = True
    elif v == 'False':
        v = False

    return k, v, v_type

############################################################
# Hook for calculation of "standing" statistics for BN lrs #
# Net. should be run over the validation set in train mode #
############################################################

class StatsCalcHook(object):
    def __init__(self):
        self.num_iter = 0
    
    def update_stats(self, module):
        for stats_name in ['mean', 'var']:
            batch_stats = getattr(module, f'running_{stats_name}')
            accum_stats = getattr(module, f'accumulated_{stats_name}')
            accum_stats = accum_stats + batch_stats
            setattr(module, f'accumulated_{stats_name}', accum_stats)
        
        self.num_iter += 1

    def remove(self, module):
        for stats_name in ['mean', 'var']:
            accum_stats = getattr(module, f'accumulated_{stats_name}') / self.num_iter
            delattr(module, f'accumulated_{stats_name}')
            getattr(module, f'running_{stats_name}').data = accum_stats

    def __call__(self, module, inputs, outputs):
        self.update_stats(module)

    @staticmethod
    def apply(module):
        for k, hook in module._forward_hooks.items():
            if isinstance(hook, StatsCalcHook):
                raise RuntimeError("Cannot register two calc_stats hooks on "
                                   "the same module")
                
        fn = StatsCalcHook()
        
        stats = getattr(module, 'running_mean')
        for stats_name in ['mean', 'var']:
            attr_name = f'accumulated_{stats_name}'
            if hasattr(module, attr_name): 
                delattr(module, attr_name)
            module.register_buffer(attr_name, torch.zeros_like(stats))

        module.register_forward_hook(fn)
        
        return fn


def stats_calculation(module):
    if 'BatchNorm' in module.__class__.__name__:
        StatsCalcHook.apply(module)

    return module

def remove_stats_calculation(module):
    if 'BatchNorm' in module.__class__.__name__:
        for k, hook in module._forward_hooks.items():
            if isinstance(hook, StatsCalcHook):
                hook.remove(module)
                del module._forward_hooks[k]
                return module

    return module

############################################################
# Spectral normalization                                   #
# Can be applied recursively (compared to PyTorch version) #
############################################################

def spectral_norm(module, name='weight', apply_to=['conv2d'], n_power_iterations=1, eps=1e-12):
    # Apply only to modules in apply_to list
    module_name = module.__class__.__name__.lower()
    if module_name not in apply_to or 'adaptive' in module_name:
        return module

    if isinstance(module, nn.ConvTranspose2d):
        dim = 1
    else:
        dim = 0

    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)

    return module

def remove_spectral_norm(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break

    return module

############################################################
# Weight averaging (both EMA and plain averaging)          #
# -- works in a similar way to spectral_norm in PyTorch    #
# -- has to be applied and removed AFTER spectral_norm     #
# -- can be applied via .apply method of any nn.Module     #
############################################################

class WeightAveragingHook(object):
    # Mode can be either "running_average" with momentum
    # or "average" for direct averaging
    def __init__(self, name='weight', mode='running_average', momentum=0.9999):
        self.name = name
        self.mode = mode
        self.momentum = momentum # running average parameter
        self.num_iter = 1 # average parameter

    def update_param(self, module):
        # Only update average values
        param = getattr(module, self.name)
        param_avg = getattr(module, self.name + '_avg')
        with torch.no_grad():
            if self.mode == 'running_average':
                param_avg.data = param_avg.data * self.momentum + param.data * (1 - self.momentum)
            elif self.mode == 'average':
                param_avg.data = (param_avg.data * self.num_iter + param.data) / (self.num_iter + 1)
                self.num_iter += 1

    def remove(self, module):
        param_avg = getattr(module, self.name + '_avg')
        delattr(module, self.name)
        delattr(module, self.name + '_avg')
        module.register_parameter(self.name, nn.Parameter(param_avg))

    def __call__(self, module, grad_input, grad_output):
        if module.training: 
            self.update_param(module)

    @staticmethod
    def apply(module, name, mode, momentum):
        for k, hook in module._forward_hooks.items():
            if isinstance(hook, WeightAveragingHook) and hook.name == name:
                raise RuntimeError("Cannot register two weight_averaging hooks on "
                                   "the same parameter {}".format(name))
                
        fn = WeightAveragingHook(name, mode, momentum)
        
        if name in module._parameters:
            param = module._parameters[name].data
        else:
            param = getattr(module, name)

        module.register_buffer(name + '_avg', param.clone())

        module.register_backward_hook(fn)
        
        return fn

class WeightAveragingPreHook(object):
    # Mode can be either "running_average" with momentum
    # or "average" for direct averaging
    def __init__(self, name='weight'):
        self.name = name
        self.spectral_norm = True
        self.enable = False

    def __call__(self, module, inputs):
        if self.enable or not module.training:
            setattr(module, self.name, getattr(module, self.name + '_avg'))

        elif not self.spectral_norm:
            setattr(module, self.name, getattr(module, self.name + '_orig') + 0) # +0 converts a parameter to a tensor with grad fn

    @staticmethod
    def apply(module, name):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightAveragingPreHook) and hook.name == name:
                raise RuntimeError("Cannot register two weight_averaging hooks on "
                                   "the same parameter {}".format(name))
                
        fn = WeightAveragingPreHook(name)

        if not hasattr(module, name + '_orig'):
            param = module._parameters[name]

            delattr(module, name)
            module.register_parameter(name + '_orig', param)
            setattr(module, name, param.data)

            fn.spectral_norm = False

        module.register_forward_pre_hook(fn)
        
        return fn


def weight_averaging(module, names=['weight', 'bias'], mode='running_average', momentum=0.9999):
    for name in names:
        if hasattr(module, name) and getattr(module, name) is not None:
            WeightAveragingHook.apply(module, name, mode, momentum)
            WeightAveragingPreHook.apply(module, name)

    return module


def remove_weight_averaging(module, names=['weight', 'bias']):
    for name in names:               
        for k, hook in module._backward_hooks.items():
            if isinstance(hook, WeightAveragingHook) and hook.name == name:
                hook.remove(module)
                del module._backward_hooks[k]
                break

        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightAveragingPreHook) and hook.name == name:
                hook.remove(module)
                del module._forward_pre_hooks[k]
                break

    return module

#############################################################
#          Postprocessing of modules for inference          #
#############################################################

def prepare_for_mobile_inference(module):
    mod = module

    if isinstance(module, nn.InstanceNorm2d) or isinstance(module, nt_utils.AdaptiveNorm2d) and isinstance(module.norm_layer, nn.InstanceNorm2d):
        # Split affine part of instance norm into separable 1x1 conv
        new_mod_1 = nn.InstanceNorm2d(module.num_features, eps=module.eps, affine=False)
        
        weight_data = module.weight.data.squeeze().detach().clone()
        bias_data = module.bias.data.squeeze().detach().clone()
        
        new_mod_2 = nn.Conv2d(module.num_features, module.num_features, 1, groups=module.num_features)
        
        new_mod_2.weight.data = weight_data.view(module.num_features, 1, 1, 1)
        new_mod_2.bias.data = bias_data

        mod = nn.Sequential(
            new_mod_1,
            new_mod_2)

    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nt_utils.AdaptiveNorm2d) and isinstance(module.norm_layer, nn.BatchNorm2d):
        mod = nn.Conv2d(module.num_features, module.num_features, 1, groups=module.num_features)

        if isinstance(module, nt_utils.AdaptiveNorm2d):
            sigma = (module.norm_layer.running_var + module.norm_layer.eps)**0.5
            mu = module.norm_layer.running_mean
        else:
            sigma = (module.running_var + module.eps)**0.5
            mu = module.running_mean
            
        sigma = sigma.clone()
        mu = mu.clone()

        gamma = module.weight.data.squeeze().detach().clone()
        beta = module.bias.data.squeeze().detach().clone()
        
        mod.weight.data[:, 0, 0, 0] = gamma / sigma
        mod.bias.data = beta - mu / sigma * gamma

    elif isinstance(module, nt_utils.AdaptiveConv2d):
        mod = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, bias=True)
        mod.weight.data = module.weight.data[0].detach().clone()
        mod.bias.data = module.bias.data[0].detach().clone()

    else:
        for name, child in module.named_children():
            mod.add_module(name, prepare_for_mobile_inference(child))

    del module
    return mod