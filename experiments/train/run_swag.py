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

parser = argparse.ArgumentParser()
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
    default="WideResNet28x10",
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
    default=300,
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

parser.add_argument("--sam", action="store_true", help="Using SAM to train model")
parser.add_argument("--sam_adaptive", action="store_true", help="Using ASAM to train model")
parser.add_argument("--rho", type=float, default=0.05, help="Using SAM to train model with rho")

parser.add_argument("--cosine_schedule", action="store_true", help="Using cosine_schedule for learning rate")
parser.add_argument("--mix_sgd", action="store_true", help="Training whole model with sgd setting")
parser.add_argument("--use_sgd", action="store_true", help="Training with sgd but through SAM optimizer to eval sharpness")
parser.add_argument("--eval_sharpness", action="store_true", help="Eval sharpness only")

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

print("Preparing model")
print(*model_cfg.args)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

if args.cov_mat:
    args.no_cov_mat = False
else:
    args.no_cov_mat = True

if args.swa:
    print("SWAG training")
    swag_model = SWAG(
        model_cfg.base,
        no_cov_mat=args.no_cov_mat,
        max_num_models=args.max_num_models,
        *model_cfg.args,
        num_classes=num_classes,
        **model_cfg.kwargs
    )
    swag_model.to(args.device)
    fake_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    fake_model.to(args.device)
else:
    print("SGD training")


def schedule(epoch):
    if not args.cosine_schedule:
        t = (epoch) / (args.swa_start if args.swa and not args.mix_sgd else args.epochs)
        lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
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


# use a slightly modified loss function that allows input of model
if args.loss == "CE":
    criterion = losses.cross_entropy
    # criterion = F.cross_entropy
elif args.loss == "adv_CE":
    criterion = losses.adversarial_cross_entropy

if args.sam:
    # import pdb; pdb.set_trace()
    optimizer = SAM(model, base_optimizer=torch.optim.SGD, rho=args.rho, lr=args.lr_init,
                    momentum=args.momentum, weight_decay=args.wd, adaptive=args.sam_adaptive, is_sgd=args.use_sgd)
    optimizer_fake_model = SAM(fake_model, base_optimizer=torch.optim.SGD, rho=args.rho, lr=args.lr_init,
                               momentum=args.momentum, weight_decay=args.wd, adaptive=args.sam_adaptive,
                               is_sgd=args.use_sgd)
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

if args.swa and args.swa_resume is not None:
    checkpoint = torch.load(args.swa_resume)
    swag_model = SWAG(
        model_cfg.base,
        no_cov_mat=args.no_cov_mat,
        max_num_models=args.max_num_models,
        loading=True,
        *model_cfg.args,
        num_classes=num_classes,
        **model_cfg.kwargs
    )
    swag_model.to(args.device)
    swag_model.load_state_dict(checkpoint["state_dict"])

columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time", "mem_usage"]
if args.swa:
    columns = columns[:-2] + ["swa_te_loss", "swa_te_acc", "swa_nll_loss", "swa_ece"] + columns[-2:]
    swag_res = {"loss": None, "accuracy": None, 'nll_loss': None, 'ece_score': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict(),
)

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0.0

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)

    train_res = utils.train_epoch(loaders["train"], model, criterion, optimizer, cuda=use_cuda)

    if (
            epoch == 0
            or epoch % args.eval_freq == args.eval_freq - 1
            or epoch == args.epochs - 1
    ):
        test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda, num_classes=num_classes)
    else:
        test_res = {"loss": None, "accuracy": None}

    if (
            args.swa
            and (epoch + 1) > args.swa_start
            and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0
    ):
        # sgd_preds, sgd_targets = utils.predictions(loaders["test"], model)
        sgd_res = utils.predict(loaders["test"], model)
        sgd_preds = sgd_res["predictions"]
        sgd_targets = sgd_res["targets"]
        if sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            sgd_ens_preds = sgd_ens_preds * n_ensembled / (
                    n_ensembled + 1
            ) + sgd_preds / (n_ensembled + 1)
        n_ensembled += 1
        swag_model.collect_model(model)  # update swag-diag and swag
        if (
                epoch == 0
                or epoch % args.eval_freq == args.eval_freq - 1
                or epoch == args.epochs - 1
        ):
            if args.eval_sharpness:
                # eval sharpness of swag_model
                num_n = 1
                swag_sharp_score = {
                    True: {},
                    False: {}
                }
                for sample_with_cov, scale in zip([True, False], [0.1, 0.2]):
                    # for sample_with_cov, scale in zip([True], [0.1]):
                    swag_sharpness_ems = {
                        "sharpness": 0.0,
                        "sharpness_abs": 0.0,
                        "grad_loss": 0.0,
                        "grad_max": 0.0,
                        "grad_flat": 0.0,
                    }
                    print("Model with sample_with_cov={}, scale={}".format(sample_with_cov, scale))
                    for i in range(num_n):
                        swag_model.sample(scale=scale, cov=sample_with_cov)
                        utils.bn_update(loaders["train"], swag_model)
                        fake_model = swag_model.transfer_model(fake_model)
                        utils.bn_update(loaders["train"], fake_model)
                        # swag_sharpness = utils.eval_sharpness(loaders["test"], fake_model, criterion, optimizer_fake_model)
                        # fake_model_res = utils.eval(loaders["test"], fake_model, criterion, cuda=use_cuda, num_classes=num_classes, data_split=10)
                        swag_sharpness = utils.eval_sharpness(loaders["train"], fake_model, criterion,
                                                              optimizer_fake_model)
                        print("\tModel idx", i)
                        print("\t\t loss:", swag_sharpness['loss'], "\t Acc:", swag_sharpness['accuracy'])
                        print("\t\t", swag_sharpness)
                        for k in swag_sharpness.keys():
                            if not swag_sharpness_ems.__contains__(k):
                                swag_sharpness_ems[k] = 0.0
                            swag_sharpness_ems[k] += swag_sharpness[k]
                    for k in swag_sharpness_ems.keys():
                        swag_sharpness_ems[k] = swag_sharpness_ems[k] / num_n
                    swag_sharp_score[sample_with_cov] = swag_sharpness_ems
                print(swag_sharp_score)
            else:
                # eval swag_model
                swag_model.sample(0.0)
                utils.bn_update(loaders["train"], swag_model)
                swag_res = utils.eval(loaders["test"], swag_model, criterion, nll=True, num_classes=num_classes)
        else:
            swag_res = {"loss": None, "accuracy": None, 'nll_loss': None, 'ece_score': None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        if args.swa:
            utils.save_checkpoint(
                args.dir, epoch + 1, name="swag", state_dict=swag_model.state_dict()
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
    if args.swa:
        values = values[:-2] + [swag_res["loss"], swag_res["accuracy"], swag_res["nll_loss"],
                                swag_res["ece_score"]] + values[-2:]

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
    if args.swa and args.epochs > args.swa_start:
        utils.save_checkpoint(
            args.dir, args.epochs, name="swag", state_dict=swag_model.state_dict()
        )

if args.swa:
    np.savez(
        os.path.join(args.dir, "sgd_ens_preds.npz"),
        predictions=sgd_ens_preds,
        targets=sgd_targets,
    )
