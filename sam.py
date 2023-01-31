import torch
import random
from torch.nn.modules.batchnorm import _BatchNorm
import contextlib
import math


def disable_running_stats(model, backup=True):
    def _disable(module):
        # if isinstance(module, _BatchNorm) and not hasattr(module, "backup_momentum"):
        #     module.backup_momentum = module.momentum
        # if isinstance(module, _BatchNorm) and module.momentum != 0:
        #     module.backup_momentum = module.momentum
        # module.momentum = 0
        if isinstance(module, _BatchNorm):
            if not hasattr(module, "backup_momentum"):
                module.backup_momentum = module.momentum
            # print("_disable", module.momentum)
            module.momentum = 0
        # elif hasattr(module, 'momentum'):
        #     print("NOTBN:", module, module.momentum)
    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            # print("_enable", module.momentum)
            module.momentum = module.backup_momentum

    model.apply(_enable)


class SAM(torch.optim.Optimizer):
    def __init__(self, model, base_optimizer, rho=0.05, adaptive=False, is_sgd=False, is_sgld=False, start_step=0, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(model.parameters(), defaults)
        self.model = model
        self.is_sgd = is_sgd
        if is_sgd:
            print("Using SAM to train model by SGD")
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.debug = False
        self.is_sgld = is_sgld  # a = 10, b = 1000, y = 0.9
        self.running_step = start_step
        # self.get_noise = lambda step: 10/(math.pow(1000 + step, 0.9))
        # self.get_noise = lambda step: 40.304/(math.pow(15476.4 + step, 0.9))  # preresnet-164
        self.get_noise = lambda step: 38.0348/(math.pow(13928.7 + step, 0.9))  # WideResnet28x10

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["old_grad"] = p.grad.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False, reverse=False, get_gradnorm=False):
        if self.debug:
            if reverse:
                print("Reverse to theta")
            if get_gradnorm:
                print("get_gradnorm is on")
        shared_device = self.param_groups[0]["params"][0].device
        g_loss_norm = torch.tensor(0.0, device=shared_device)
        g_max_norm = torch.tensor(0.0, device=shared_device)
        g_flat_norm = torch.tensor(0.0, device=shared_device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # if p.name is None:
                #     print(p)
                # if p.name is not None and ('cov_mat_sqrt' in p.name or "_mean" in p.name): continue
                # else:
                #     print(p.name, p)
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                if get_gradnorm:
                    g_loss_norm += torch.mul(self.state[p]["old_grad"], self.state[p]["old_grad"]).sum()
                    g_max_norm += torch.mul(p.grad.data, p.grad.data).sum()
                    g_flat = p.grad.data - self.state[p]["old_grad"]
                    g_flat_norm += torch.mul(g_flat, g_flat).sum()
                if reverse:
                    p.grad.data = self.state[p]["old_grad"]
                if self.is_sgld:
                    epsilon = self.get_noise(self.running_step)
                    noise = 5e-4 * epsilon * torch.randn_like(p.data)
                    # p.grad.data = epsilon/2 * p.grad.data + noise
                    # self.state[p]["noise"] = epsilon/2 * p.grad.data + noise
                    self.state[p]["noise"] = noise
                    # p.grad.add_(noise)
        if get_gradnorm:
            g_loss_norm = torch.sqrt(g_loss_norm)
            g_max_norm = torch.sqrt(g_max_norm)
            g_flat_norm = torch.sqrt(g_flat_norm)
        # self.base_optimizer.step()  # do the actual "sharpness-aware" update

        # if zero_grad: self.zero_grad()
        return g_loss_norm, g_max_norm, g_flat_norm
    @torch.no_grad()
    def step(self, get_gradnorm=False, update_model=True):
        # self.forward_backward_func()  # true loss function
        if self.debug:
            if not update_model:
                print("NOT update model and Reverse to theta")
        with self.maybe_no_sync():
            self.first_step(zero_grad=True)
            disable_running_stats(self.model)
            outputs_max, loss_value_max = self.forward_backward_func()
            g_loss_norm, g_max_norm, g_flat_norm = self.second_step(reverse=self.is_sgd or not update_model, get_gradnorm=get_gradnorm)
        self._sync_grad()
        if update_model:
            # if self.is_sgld:
            #     for group in self.param_groups:
            #         for p in group["params"]:
            #             if p.grad is None: continue
            #             lr_sgld = self.get_noise(self.running_step)
            #             group['lr'] = lr_sgld
            #             # noise = lr_sgld*torch.randn_like(p.data)
            #             # p.data.add_(-noise)
            self.base_optimizer.step()
            if self.is_sgld:
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None: continue
                        # noise = 0.1*group['lr']*torch.randn_like(p.data)
                        p.data.add_(self.state[p]["noise"])
            self.base_optimizer.zero_grad()
        self.zero_grad()
        enable_running_stats(self.model)
        self.running_step += 1
        return outputs_max, loss_value_max, g_loss_norm, g_max_norm, g_flat_norm

    @torch.no_grad()
    def set_closure(self, loss_fn, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                # outputs = self.model(inputs)
                loss, outputs = loss_fn(**kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
class SGD_Cus(torch.optim.Optimizer):
    def __init__(self, model, base_optimizer, is_sgld=False, **kwargs):
        defaults = dict(rho=0, adaptive=False, **kwargs)
        super(SGD_Cus, self).__init__(model.parameters(), defaults)
        self.model = model
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.debug = False
        self.is_sgld = is_sgld  # a = 10, b = 1000, y = 0.9
        self.running_step = 0
        self.get_noise = lambda step: 10/(math.pow(1000 + step, 0.9))
    @torch.no_grad()
    def step(self, get_gradnorm=False, update_model=True):
        if update_model:
            if self.is_sgld:
                for group in self.param_groups:
                    lr_sgld = self.get_noise(self.running_step)
                    group['lr'] = lr_sgld

                        # noise = lr_sgld*torch.randn_like(p.data)
                        # p.data.add_(-noise)
            self.base_optimizer.step()
            if self.is_sgld:
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None: continue
                        noise = 0.1*group['lr']*torch.randn_like(p.data)
                        p.data.add_(-noise)
            self.base_optimizer.zero_grad()
        self.zero_grad()
        self.running_step += 1
