import argparse
import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import math

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
sys.stdout = utils.Logger(log_path=args.dir, fine_tune=args.resume)


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

def schedule(epoch):
    if not args.cosine_schedule:
        t = epoch / args.epochs
        lr_ratio = 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return args.lr_init * factor
    else:
        lr_min = args.lr_init / 1000
        if epoch % 2 == 0:
            lr = lr_min + (args.lr_init - lr_min) / 2 * (1 + math.cos(math.pi * epoch / args.epochs))
        else:
            lr_prev = lr_min + (args.lr_init - lr_min) / 2 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs))
            lr = lr_prev + (args.lr_init - lr_min) / 2 * (1 - math.cos(math.pi / args.epochs))
        # print(lr)
        return lr

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

ems_prediction = None
for model_idx in range(args.max_num_models):

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(model_idx)

    if use_cuda:
        torch.cuda.manual_seed(model_idx)

    print("Preparing model {}".format(model_idx))
    print(*model_cfg.args)
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    model.to(args.device)

    if args.sam:
        optimizer = SAM(model, base_optimizer=torch.optim.SGD, rho=args.rho, lr=args.lr_init,
                        momentum=args.momentum, weight_decay=args.wd, adaptive=args.sam_adaptive)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd
        )

    start_epoch = 0
    if args.resume is not None:
        print("Resume training from %s" % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time", "mem_usage"]

    sgd_ens_preds = None
    sgd_targets = None
    n_ensembled = 0.0
    for epoch in range(start_epoch, args.epochs):
        time_ep = time.time()

        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)

        train_res = utils.train_epoch(loaders["train"], model, criterion, optimizer, cuda=use_cuda)
        if epoch % 50 == 0:
            test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda, num_classes=num_classes)

            time_ep = time.time() - time_ep

            if use_cuda:
                memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

            values = [
                epoch + 1,
                lr,
                train_res["loss"],
                train_res["accuracy"],
                test_res["loss"],
                test_res["accuracy"],
                time_ep,
                memory_usage,
            ]

            table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
            if epoch % 40 == 0:
                table = table.split("\n")
                table = "\n".join([table[1]] + table)
            else:
                table = table.split("\n")[2]
            print(table)
    # eval model idx and emsemble prediction
    test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda, num_classes=num_classes, ems_prediction=ems_prediction)
    ems_prediction = test_res['ems_prediction']
    ems_target = test_res['targets']

    eval_acc = np.mean(np.argmax(ems_prediction, axis=1) == ems_target) * 100
    _, eval_nll = nll(ems_prediction / (model_idx + 1), ems_target)
    eval_ece, acc, conf, Bm = losses.ece_score_np(ems_prediction / (model_idx + 1), ems_target, n_bins=20,
                                                  return_stat=True)
    print("====== Model {}/{} ======".format(model_idx+1, args.max_num_models))
    print("Accuracy:", eval_acc)
    print("NLL:", eval_nll)
    print("ECE:", eval_ece)
    print('--------- ECE stat ----------')
    print("ACC", acc)
    print("Conf", conf)
    print("Bm", Bm)
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
        name='checkpoint_{}'.format(model_idx)
    )
    del model
    del optimizer



# eval_acc = np.mean(np.argmax(ems_prediction, axis=1) == ems_target) * 100
# _, eval_nll = nll(ems_prediction / args.max_num_models, ems_target)
# eval_ece, acc, conf, Bm = losses.ece_score_np(ems_prediction / args.max_num_models, ems_target, n_bins=20, return_stat=True)
# print("Accuracy:", eval_acc)
# print("NLL:", eval_nll)
# print("ECE:", eval_ece)
# print('--------- ECE stat ----------')
# print("ACC", acc)
# print("Conf", conf)
# print("Bm", Bm)
