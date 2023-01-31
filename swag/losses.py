import torch
import torch.nn.functional as F
import numpy as np


def cross_entropy(model, input, target, nll=False):
    # standard cross-entropy loss function

    output = model(input)

    loss = F.cross_entropy(output, target)
    if nll:
        swa_predictions = F.softmax(output, dim=-1)
        idx = torch.arange(len(swa_predictions))
        nll_loss = -torch.log(swa_predictions[idx, target] + 1e-12).mean()
        ece_loss = ece_score(swa_predictions, target)
        return loss, output, nll_loss, ece_loss
    return loss, output


def adversarial_cross_entropy(
    model, input, target, lossfn=F.cross_entropy, epsilon=0.01, nll=False
):
    # loss function based on algorithm 1 of "simple and scalable uncertainty estimation using
    # deep ensembles," lakshminaraynan, pritzel, and blundell, nips 2017,
    # https://arxiv.org/pdf/1612.01474.pdf
    # note: the small difference bw this paper is that here the loss is only backpropped
    # through the adversarial loss rather than both due to memory constraints on preresnets
    # we can change back if we want to restrict ourselves to VGG-like networks (where it's fine).

    # scale epsilon by min and max (should be [0,1] for all experiments)
    # see algorithm 1 of paper
    scaled_epsilon = epsilon * (input.max() - input.min())

    # force inputs to require gradient
    input.requires_grad = True

    # standard forwards pass
    output = model(input)
    loss = lossfn(output, target)

    # now compute gradients wrt input
    loss.backward(retain_graph=True)

    # now compute sign of gradients
    inputs_grad = torch.sign(input.grad)

    # perturb inputs and use clamped output
    inputs_perturbed = torch.clamp(
        input + scaled_epsilon * inputs_grad, 0.0, 1.0
    ).detach()
    # inputs_perturbed.requires_grad = False

    input.grad.zero_()
    # model.zero_grad()

    outputs_perturbed = model(inputs_perturbed)

    # compute adversarial version of loss
    adv_loss = lossfn(outputs_perturbed, target)

    if nll:
        swa_predictions = F.softmax(output, dim=-1)
        idx = torch.arange(len(swa_predictions))
        nll_loss = -torch.log(swa_predictions[idx, target] + 1e-12).mean()
        ece_loss = ece_score(swa_predictions, target)
        return (loss + adv_loss) / 2.0, output, nll_loss, ece_loss
    # return mean of loss for reasonable scalings
    return (loss + adv_loss) / 2.0, output


def masked_loss(y_pred, y_true, void_class=11.0, weight=None, reduce=True):
    # masked version of crossentropy loss

    el = torch.ones_like(y_true) * void_class
    mask = torch.ne(y_true, el).long()

    y_true_tmp = y_true * mask

    loss = F.cross_entropy(y_pred, y_true_tmp, weight=weight, reduction="none")
    loss = mask.float() * loss

    if reduce:
        return loss.sum() / mask.sum()
    else:
        return loss, mask


def seg_cross_entropy(model, input, target, weight=None):
    output = model(input)

    # use masked loss function
    loss = masked_loss(output, target, weight=weight)

    return {"loss": loss, "output": output}


def seg_ale_cross_entropy(model, input, target, num_samples=50, weight=None):
    # requires two outputs for model(input)

    output = model(input)
    mean = output[:, 0, :, :, :]
    scale = output[:, 1, :, :, :].abs()

    output_distribution = torch.distributions.Normal(mean, scale)

    total_loss = 0

    for _ in range(num_samples):
        sample = output_distribution.rsample()

        current_loss, mask = masked_loss(sample, target, weight=weight, reduce=False)
        total_loss = total_loss + current_loss.exp()
    mean_loss = total_loss / num_samples

    return {"loss": mean_loss.log().sum() / mask.sum(), "output": mean, "scale": scale}


def ece_score(py, y_test, n_bins=20):
    """

    Args:
        py: softmax prediction of model
        y_test: target (onehot or label)
        n_bins: number of bins

    Returns:

    """
    if y_test.ndim > 1:
        y_test = torch.argmax(y_test, axis=1)
    py_index = torch.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = torch.Tensor(py_value)
    acc, conf = torch.zeros(n_bins), torch.zeros(n_bins)
    Bm = torch.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * torch.abs((acc[m] - conf[m]))
    return ece / Bm.sum()

def ece_score_np(py, y_test, n_bins=20, return_stat=False):
    """

    Args:
        py: softmax prediction of model
        y_test: target (onehot or label)
        n_bins: number of bins

    Returns:

    """
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.asarray(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    if return_stat:
        return ece / sum(Bm), acc, conf, Bm
    return ece / sum(Bm)
