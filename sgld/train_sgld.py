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

from sam import SAM, SGD_Cus

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--dir",
    type=str,
    default=None,
    required=True,
    help="training directory (default: None)",
)

parser.add_argument(
    "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
)
parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    required=True,
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--use_test",
    dest="use_test",
    action="store_true",
    help="use test dataset instead of validation (default: False)",
)
parser.add_argument("--split_classes", type=int, default=None)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    metavar="MODEL",
    help="model name (default: None)",
)

parser.add_argument(
    "--resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to resume training from (default: None)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 200)",
)
parser.add_argument(
    "--save_freq",
    type=int,
    default=25,
    metavar="N",
    help="save frequency (default: 25)",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=5,
    metavar="N",
    help="evaluation frequency (default: 5)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.01,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)

parser.add_argument("--swa", action="store_true", help="swa usage flag (default: off)")
parser.add_argument(
    "--swa_start",
    type=float,
    default=161,
    metavar="N",
    help="SWA start epoch number (default: 161)",
)
parser.add_argument(
    "--swa_lr", type=float, default=0.02, metavar="LR", help="SWA LR (default: 0.02)"
)
parser.add_argument(
    "--swa_c_epochs",
    type=int,
    default=1,
    metavar="N",
    help="SWA model collection frequency/cycle length in epochs (default: 1)",
)
parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")
parser.add_argument(
    "--max_num_models",
    type=int,
    default=20,
    help="maximum number of SWAG models to save",
)

parser.add_argument(
    "--swa_resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to restor SWA from (default: None)",
)
parser.add_argument(
    "--loss",
    type=str,
    default="CE",
    help="loss to use for training model (default: Cross-entropy)",
)

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument("--no_schedule", action="store_true", help="store schedule")

parser.add_argument("--sam", action="store_true", help="Using SAM to train model")
parser.add_argument("--sam_adaptive", action="store_true", help="Using ASAM to train model")
parser.add_argument("--rho", type=float, default=0.05, help="Using SAM to train model with rho")

parser.add_argument("--cosine_schedule", action="store_true", help="Using cosine_schedule for learning rate")
parser.add_argument("--mix_sgd", action="store_true", help="Training whole model with sgd setting")
parser.add_argument("--use_sgd", action="store_true", help="Training with sgd but through SAM optimizer")
parser.add_argument("--eval_sharpness", action="store_true", help="Eval sharpness only")
parser.add_argument("--fine_tune", action="store_true", help="fine_tune ")

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
sys.stdout = utils.Logger(log_path=args.dir, fine_tune=args.resume or args.swa_resume)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed(args.seed)

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

start_epoch = 0
print("Preparing model")
print(*model_cfg.args)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
checkpoint = torch.load(args.resume)
model.load_state_dict(checkpoint["state_dict"])
model.to(args.device)

if args.fine_tune:
    start_epoch = checkpoint['epoch']


def schedule(epoch):
    lr = 0.2 / math.pow(0.1 + epoch, 0.55)
    if lr > 0.2:
       lr = 0.2
    return lr
    # if not args.cosine_schedule:
    #     t = (epoch) / (args.swa_start if args.swa and not args.mix_sgd else args.epochs)
    #     lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    #     if t <= 0.5:
    #         factor = 1.0
    #     elif t <= 0.9:
    #         factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    #     else:
    #         factor = lr_ratio
    #     return args.lr_init * factor
    # else:
    #     lr_min = args.lr_init / 1000
    #     if epoch % 2 == 0:
    #         lr = lr_min + (args.lr_init - lr_min) / 2 * (1 + math.cos(math.pi * epoch / args.epochs))
    #     else:
    #         lr_prev = lr_min + (args.lr_init - lr_min) / 2 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs))
    #         lr = lr_prev + (args.lr_init - lr_min) / 2 * (1 - math.cos(math.pi / args.epochs))
    #     # print(lr)
    #     return lr


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

if args.sam:
    # import pdb; pdb.set_trace()
    optimizer = SAM(model, base_optimizer=torch.optim.SGD, rho=args.rho, is_sgld=True, lr=args.lr_init,
                    momentum=args.momentum, weight_decay=args.wd, adaptive=args.sam_adaptive, start_step=start_epoch*391)
    if args.fine_tune:
        optimizer.load_state_dict(checkpoint["optimizer"])
else:
    optimizer = SGD_Cus(model, base_optimizer=torch.optim.SGD, is_sgld=True, lr=args.lr_init,
                    momentum=args.momentum, weight_decay=args.wd)

columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time", "mem_usage"]
columns = columns[:-2] + ["sgld_te_loss", "sgld_te_acc", "sgld_nll_loss", "sgld_ece"] + columns[-2:]
# swag_res = {"loss": None, "accuracy": None, 'nll_loss': None, 'ece_score': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict(),
)

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0.0

swa_infor = {'max_acc': -1, 'cur_acc': None, 'min_nll': 10, 'cur_nll': None, 'min_ece': 100, 'cur_ece': None}

num_collected = 0
ems_prediction = None
for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)
    # if not args.no_schedule:
    #     lr = schedule(epoch)
    #     utils.adjust_learning_rate(optimizer, lr)
    # else:
    #     lr = args.lr_init
    if epoch == 0:
        train_res = {"loss": None, "accuracy": None,}
    else:
        train_res = utils.train_epoch(loaders["train"], model, criterion, optimizer, cuda=use_cuda)
    test_res = {"loss": None, "accuracy": None, "ems_prediction": None}
    # lr = optimizer.get_noise(optimizer.running_step)
    if epoch >= args.swa_start:
        test_res = utils.eval(loaders["test"], model, criterion, cuda=True, num_classes=num_classes,
                              ems_prediction=ems_prediction)
        # for k in test_res.keys():
        #     if not test_res_collect.__contains__(k):
        #         test_res_collect[k] = test_res[k]
        #     else:
        #         test_res_collect[k] = test_res_collect[k][:] + test_res[k][:]
        ems_prediction = test_res["ems_prediction"]
        num_collected += 1
        predictions = ems_prediction / num_collected
        targets = test_res["targets"]

        eval_acc = np.mean(np.argmax(predictions, axis=1) == targets) * 100
        _, eval_nll = nll(predictions, targets)
        eval_ece, acc_bin, conf_bin, Bm_bin = losses.ece_score_np(predictions, targets, n_bins=20, return_stat=True)
        # print("Accuracy:", eval_acc)
        # print("NLL:", eval_nll)
        # print("ECE:", eval_ece)
        print('--------- ECE stat ----------', np.max(ems_prediction))

        print("ACC", acc_bin)
        print("Conf", conf_bin)
        print("Bm", Bm_bin)
    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )

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
    if epoch >= args.swa_start:
        # "loss": loss_sum / num_objects_total,
        #             "accuracy": None if regression else correct / num_objects_total * 100.0,
        #             "nll_loss": nll_loss_sum / num_objects_total,
        #             "ece_score": ece_score_sum / num_objects_total,
        #             "ems_prediction": ems_prediction,
        #             "targets": targets,
        values = values[:-2] + [None, eval_acc, eval_nll,
                                eval_ece] + values[-2:]
    else:

        values = values[:-2] + [None, None, None,
                                None] + values[-2:]

    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
