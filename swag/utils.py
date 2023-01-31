import itertools
import torch
import sys
import os
import signal
import copy
from datetime import datetime
import math
import numpy as np
import tqdm

import torch.nn.functional as F
from sam import SAM

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList


def LogSumExp(x, dim=0):
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim, keepdim=True))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(dir, epoch, name="checkpoint", replace=False, **kwargs):
    state = {"epoch": epoch}
    state.update(kwargs)
    if replace:
        for root, dirs, files in os.walk(dir):
            for f_n in files:
                if name in f_n:
                    os.remove(os.path.join(root, f_n))
                    print("Remove", f_n)
    filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    torch.save(state, filepath)


def train_epoch(
    loader,
    model,
    criterion,
    optimizer,
    cuda=True,
    regression=False,
    verbose=False,
    subset=None,
):
    loss_sum = 0.0
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, output = criterion(model, input, target)

        optimizer.zero_grad()
        loss.backward()
        if isinstance(optimizer, SAM):
            optimizer.set_closure(criterion, model=model, input=input, target=target)
        optimizer.step()
        loss_sum += loss.data.item() * input.size(0)

        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print(
                "Stage %d/10. Loss: %12.4f. Acc: %6.2f"
                % (
                    verb_stage + 1,
                    loss_sum / num_objects_current,
                    correct / num_objects_current * 100.0,
                )
            )
            verb_stage += 1

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None if regression else correct / num_objects_current * 100.0,
    }


def eval(loader, model, criterion, cuda=True, regression=False, verbose=False, nll=False, ems_prediction=None, num_classes=10):
    loss_sum = 0.0
    correct = 0.0
    nll_loss_sum = 0.0
    ece_score_sum = 0.0
    num_objects_total = len(loader.dataset)
    if ems_prediction is None:
        ems_prediction = np.zeros((num_objects_total, num_classes))
    targets = np.zeros(num_objects_total)
    model.eval()
    k = 0
    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        for i, (input, target) in enumerate(loader):
            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            cre_out = criterion(model, input, target, nll=nll)
            if nll:
                loss, output, nll_loss, ece_score = cre_out
                nll_loss_sum += nll_loss.item() * input.size(0)
                ece_score_sum += ece_score.item() * input.size(0)
            else:
                loss, output = cre_out

            loss_sum += loss.item() * input.size(0)

            if not regression:
                pred = output.data.argmax(1, keepdim=True)
                correct += pred.eq(target.data.view_as(pred)).sum().item()
            bz = input.size()[0]
            ems_prediction[k: k + bz] += (F.softmax(output, dim=1).cpu().numpy())
            targets[k: (k + bz)] = target.cpu().numpy()
            k += bz
    return {
            "loss": loss_sum / num_objects_total,
            "accuracy": None if regression else correct / num_objects_total * 100.0,
            "nll_loss": nll_loss_sum / num_objects_total,
            "ece_score": ece_score_sum / num_objects_total,
            "ems_prediction": ems_prediction,
            "targets": targets,
        }
def eval_sharpness(loader, model, criterion, optimizer, cuda=True, regression=False, verbose=False, nll=False):
    grad_loss = 0.0
    grad_max = 0.0
    grad_flat = 0.0
    sharpness_abs = 0.0
    sharpness = 0.0
    data_total = len(loader.dataset)
    loss_sum = 0.0
    loss_max_sum = 0.0
    model.train()

    if verbose:
        loader = tqdm.tqdm(loader)
    num_objects_total = 0
    correct = 0
    for i, (input, target) in enumerate(loader):
        if num_objects_total > data_total/2:
            break
        bz = input.size(0)
        num_objects_total += bz
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, output = criterion(model, input, target)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.set_closure(criterion, model=model, input=input, target=target)
        outputs_max, loss_value_max, g_loss_norm, g_max_norm, g_flat_norm = optimizer.step(update_model=True, get_gradnorm=True)

        loss_max_sum += loss_value_max.item() * input.size(0)
        sharpness_abs += bz*torch.abs(loss_value_max-loss).item()
        sharpness += bz*(loss_value_max-loss).item()
        grad_loss += bz * g_loss_norm.item()
        grad_max += bz * g_max_norm.item()
        grad_flat += bz * g_flat_norm.item()

    return {
            'loss_max': loss_max_sum / num_objects_total,
            "loss": loss_sum / num_objects_total,
            "accuracy": correct / num_objects_total,
            "sharpness": sharpness / num_objects_total,
            "sharpness_abs": sharpness_abs / num_objects_total,
            "grad_loss": grad_loss / num_objects_total,
            "grad_max": grad_max / num_objects_total,
            "grad_flat": grad_flat / num_objects_total,
        }


def predict(loader, model, verbose=False):
    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    with torch.no_grad():
        for input, target in loader:
            input = input.cuda(non_blocking=True)
            output = model(input)

            batch_size = input.size(0)
            predictions.append(F.softmax(output, dim=1).cpu().numpy())
            targets.append(target.numpy())
            offset += batch_size

    return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:

            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def inv_softmax(x, eps=1e-10):
    return torch.log(x / (1.0 - x + eps))


def predictions(test_loader, model, seed=None, cuda=True, regression=False, **kwargs):
    # will assume that model is already in eval mode
    # model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        if seed is not None:
            torch.manual_seed(seed)
        if cuda:
            input = input.cuda(non_blocking=True)
        output = model(input, **kwargs)
        if regression:
            preds.append(output.cpu().data.numpy())
        else:
            probs = F.softmax(output, dim=1)
            preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def schedule(epoch, lr_init, epochs, swa, swa_start=None, swa_lr=None):
    t = (epoch) / (swa_start if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor



class Logger(object):
    def __init__(self, log_path="./logs", fine_tune=False, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.console_log_path = os.path.join(log_path, filename)
        if not os.path.exists(os.path.dirname(self.console_log_path)):
            os.makedirs(os.path.dirname(self.console_log_path))
        self.log = open(self.console_log_path, 'a' if fine_tune else 'w')
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C.')
        self.log.close()

        # Remove logfile
        # os.remove(self.console_log_path)
        # print('Removed console_output file')
        sys.exit(0)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
