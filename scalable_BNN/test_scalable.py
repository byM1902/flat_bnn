import argparse
import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import math
import tqdm

from swag import data, models, utils, losses
from swag.posteriors import SWAG

from sam import SAM

parser = argparse.ArgumentParser(description="Scalable BNN")
parser.add_argument("--dir", type=str, default=None, required=True, help="training directory (default: None)",)

parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)")
parser.add_argument("--data_path", type=str, default=None, required=True, metavar="PATH",
    help="path to datasets location (default: None)")
parser.add_argument("--use_test", dest="use_test", action="store_true",
    help="use test dataset instead of validation (default: False)")
parser.add_argument("--split_classes", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=128, metavar="N", help="input batch size (default: 128)")
parser.add_argument("--num_workers", type=int, default=4, metavar="N", help="number of workers (default: 4)")
parser.add_argument("--model", type=str, default=None, required=True, metavar="MODEL", help="model name (default: None)")

parser.add_argument("--resume", type=str, default=None, metavar="CKPT",
    help="checkpoint to resume training from (default: None)")

parser.add_argument("--epochs", type=int, default=200, metavar="N", help="number of epochs to train (default: 200)")
parser.add_argument("--save_freq", type=int, default=25, metavar="N", help="save frequency (default: 25)")
parser.add_argument("--eval_freq", type=int, default=5, metavar="N", help="evaluation frequency (default: 5)")
parser.add_argument("--lr_init", type=float, default=0.01, metavar="LR", help="initial learning rate (default: 0.01)")
parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)")
parser.add_argument("--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)")
parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")
parser.add_argument("--max_num_models", type=int, default=5,  help="maximum number of models to parallel training")

parser.add_argument("--loss",  type=str, default="CE",  help="loss to use for training model (default: Cross-entropy)")

parser.add_argument("--sam", action="store_true", help="Using SAM to train model")
parser.add_argument("--sam_adaptive", action="store_true", help="Using ASAM to train model")
parser.add_argument("--rho", type=float, default=0.05, help="Using SAM to train model with rho")

parser.add_argument("--cosine_schedule", action="store_true", help="Using cosine_schedule for learning rate")
parser.add_argument("--mix_sgd", action="store_true", help="Training whole model with sgd setting")

args = parser.parse_args()

args.device = None

use_cuda = torch.cuda.is_available()
if use_cuda:
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

for arg_item in vars(args):
    print(arg_item, getattr(args, arg_item))
print("Preparing directory %s" % args.dir)

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, "command.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")

sys.stdout = utils.Logger(log_path=args.dir, fine_tune=args.resume, filename="eval_model.txt")


print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

print("Loading dataset %s from %s" % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    split_classes=args.split_classes,
)

def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    ps_log = np.log(ps)
    nll_sum = -np.sum(ps_log)
    nll_mean = -np.mean(ps_log)
    return nll_sum, nll_mean


# use a slightly modified loss function that allows input of model
if args.loss == "CE":
    criterion = losses.cross_entropy
    # criterion = F.cross_entropy
elif args.loss == "adv_CE":
    criterion = losses.adversarial_cross_entropy

print(*model_cfg.args)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

predictions = np.zeros((len(loaders["test"].dataset), num_classes))
targets = np.zeros(len(loaders["test"].dataset))

model_lst = []
for root, dirs, files in os.walk(args.dir):
    for f_n in files:
        if "-300.pt" in f_n:
            model_lst.append(f_n)

for model_idx, ckpt_name in enumerate(model_lst):
    if model_idx == args.max_num_models :
        break
    model_path = os.path.join(args.dir, ckpt_name)
    print("Preparing model {} from {}".format(model_idx, model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    k = 0
    for input, target in tqdm.tqdm(loaders["test"]):
        input = input.cuda(non_blocking=True)

        output = model(input)

        with torch.no_grad():
            predictions[k: k + input.size()[0]] += (
                F.softmax(output, dim=1).cpu().numpy()
            )
        targets[k: (k + target.size(0))] = target.numpy()
        k += input.size()[0]

eval_acc = np.mean(np.argmax(predictions, axis=1) == targets) * 100
_, eval_nll = nll(predictions / args.max_num_models, targets)
eval_ece, acc, conf, Bm = losses.ece_score_np(predictions / args.max_num_models, targets, n_bins=20, return_stat=True)
print("Accuracy:", eval_acc)
print("NLL:", eval_nll)
print("ECE:", eval_ece)
print('--------- ECE stat ----------')
print("ACC", acc)
print("Conf", conf)
print("Bm", Bm)
