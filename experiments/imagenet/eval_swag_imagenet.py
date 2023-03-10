import argparse
import os
import random
import sys
import time
import tabulate

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.models

import data
from swag import utils, losses
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(description="SGD/SWA training")

parser.add_argument("--data_path", type=str, default=None, required=True, metavar="PATH",
                    help="path to datasets location (default: None)",)
parser.add_argument("--batch_size", type=int, default=256, metavar="N", help="input batch size (default: 256)",)
parser.add_argument("--num_workers", type=int, default=4, metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument("--model", type=str, default=None, required=True, metavar="MODEL",
    help="model name (default: None)",
)

parser.add_argument("--ckpt", type=str, required=True, default=None, metavar="CKPT",
    help="checkpoint to load (default: None)",
)
parser.add_argument("--ckpt_sgd", type=str, required=True, default=None, metavar="CKPT",
    help="checkpoint to load checkpoint-10.pt",
)

parser.add_argument("--num_samples", type=int, default=30, metavar="N",
    help="number of samples for SWAG (default: 30)",
)

parser.add_argument("--scale", type=float, default=0.5, help="SWAG scale")
parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")
parser.add_argument(
    "--use_diag_bma", action="store_true", help="sample only diag variacne for BMA"
)

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)

parser.add_argument("--save_path_swa", type=str, default=None, help="path to SWA npz results file",)
parser.add_argument("--save_path_swag", type=str, default=None, help="path to SWAG npz results file",)

args = parser.parse_args()
root_path = os.path.dirname(args.ckpt)
sys.stdout = utils.Logger(log_path=root_path, fine_tune=True)

for arg_item in vars(args):
    print(arg_item, getattr(args, arg_item))
eps = 1e-12

args.device = None
if torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("Using model %s" % args.model)
model_class = getattr(torchvision.models, args.model)
print("Loading ImageNet from %s" % (args.data_path))
loaders, num_classes = data.loaders(args.data_path, args.batch_size, args.num_workers)

print("Preparing model for SGD")
model = model_class(num_classes=num_classes)
model.to(args.device)
print("Loading checkpoint %s" % args.ckpt_sgd)
checkpoint = torch.load(args.ckpt_sgd)
model.load_state_dict(checkpoint["state_dict"])

test_res = utils.predict(loaders["test"], model, verbose=True)

targets = test_res["targets"]
test_predictions = test_res["predictions"]

swa_accuracy = np.mean(np.argmax(test_predictions, axis=1) == targets)
swa_nll = -np.mean(
    np.log(test_predictions[np.arange(test_predictions.shape[0]), targets] + eps)
)

swa_ece = losses.ece_score_np(test_predictions, targets, n_bins=20)
print("SGD. Accuracy: %.2f%% NLL: %.4f ECE: %.4f" % (swa_accuracy * 100, swa_nll, swa_ece))

del model

print("Preparing model for SWA")
swag_model = SWAG(
    model_class,
    no_cov_mat=not args.cov_mat,
    # loading=True,
    max_num_models=20,
    num_classes=num_classes,
)
swag_model.to(args.device)

criterion = losses.cross_entropy

print("Loading checkpoint %s" % args.ckpt)
checkpoint = torch.load(args.ckpt)
swag_model.load_state_dict(checkpoint["state_dict"])

print("SWA")
swag_model.sample(0.0)
# print("SWA BN update")
utils.bn_update(loaders["train"], swag_model, verbose=True, subset=0.1)
print("SWA EVAL")
swa_res = utils.predict(loaders["test"], swag_model, verbose=True)

targets = swa_res["targets"]
swa_predictions = swa_res["predictions"]

swa_accuracy = np.mean(np.argmax(swa_predictions, axis=1) == targets)
swa_nll = -np.mean(
    np.log(swa_predictions[np.arange(swa_predictions.shape[0]), targets] + eps)
)

swa_ece = losses.ece_score_np(swa_predictions, targets, n_bins=20)
print("SWA. Accuracy: %.2f%% NLL: %.4f ECE: %.4f" % (swa_accuracy * 100, swa_nll, swa_ece))
# swa_entropies = -np.sum(np.log(swa_predictions + eps) * swa_predictions, axis=1)

# np.savez(
#     args.save_path_swa,
#     accuracy=swa_accuracy,
#     nll=swa_nll,
#     entropies=swa_entropies,
#     predictions=swa_predictions,
#     targets=targets,
# )
args.use_diag_bma = False
swag_scale = args.scale

print("SWAG - scale", args.scale)
swag_predictions = np.zeros((len(loaders["test"].dataset), num_classes))

for i in range(args.num_samples):
    swag_model.sample(swag_scale, cov=args.cov_mat and (not args.use_diag_bma))

    # print("SWAG Sample %d/%d. BN update" % (i + 1, args.num_samples))
    utils.bn_update(loaders["train"], swag_model, verbose=True, subset=0.1)
    # print("SWAG Sample %d/%d. EVAL" % (i + 1, args.num_samples))
    res = utils.predict(loaders["test"], swag_model, verbose=True)
    predictions = res["predictions"]

    accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
    nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + eps))
    print(
        "\tSWAG Sample %d/%d. Accuracy: %.2f%% NLL: %.4f"
        % (i + 1, args.num_samples, accuracy * 100, nll)
    )

    swag_predictions += predictions

    ens_accuracy = np.mean(np.argmax(swag_predictions, axis=1) == targets)
    ens_nll = -np.mean(
        np.log(
            swag_predictions[np.arange(swag_predictions.shape[0]), targets] / (i + 1)
            + eps
        )
    )

    eval_ece = losses.ece_score_np(swag_predictions / (i + 1), targets, n_bins=20)
    print(
        "Ensemble %d/%d. Accuracy: %.2f%% NLL: %.4f  ECE: %.4f"
        % (i + 1, args.num_samples, ens_accuracy * 100, ens_nll, eval_ece)
    )

swag_predictions /= args.num_samples

swag_accuracy = np.mean(np.argmax(swag_predictions, axis=1) == targets)
swag_nll = -np.mean(
    np.log(swag_predictions[np.arange(swag_predictions.shape[0]), targets] + eps)
)

eval_ece = losses.ece_score_np(swag_predictions, targets, n_bins=20)
swag_entropies = -np.sum(np.log(swag_predictions + eps) * swag_predictions, axis=1)

# np.savez(
#     args.save_path_swag,
#     accuracy=swag_accuracy,
#     nll=swag_nll,
#     entropies=swag_entropies,
#     predictions=swag_predictions,
#     targets=targets,
# )

print("SWAG-Diag scale", args.scale*2)

args.use_diag_bma = False
swag_scale = args.scale*2
swag_predictions = np.zeros((len(loaders["test"].dataset), num_classes))

for i in range(args.num_samples):
    swag_model.sample(swag_scale, cov=args.cov_mat and (not args.use_diag_bma))

    # print("SWAG Sample %d/%d. BN update" % (i + 1, args.num_samples))
    utils.bn_update(loaders["train"], swag_model, verbose=True, subset=0.1)
    # print("SWAG Sample %d/%d. EVAL" % (i + 1, args.num_samples))
    res = utils.predict(loaders["test"], swag_model, verbose=True)
    predictions = res["predictions"]

    accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
    nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + eps))
    print(
        "\tSWAG-Diag Sample %d/%d. Accuracy: %.2f%% NLL: %.4f"
        % (i + 1, args.num_samples, accuracy * 100, nll)
    )

    swag_predictions += predictions

    ens_accuracy = np.mean(np.argmax(swag_predictions, axis=1) == targets)
    ens_nll = -np.mean(
        np.log(
            swag_predictions[np.arange(swag_predictions.shape[0]), targets] / (i + 1)
            + eps
        )
    )

    eval_ece = losses.ece_score_np(swag_predictions / (i + 1), targets, n_bins=20)
    print(
        "Ensemble %d/%d. Accuracy: %.2f%% NLL: %.4f  ECE: %.4f"
        % (i + 1, args.num_samples, ens_accuracy * 100, ens_nll, eval_ece)
    )

swag_predictions /= args.num_samples

swag_accuracy = np.mean(np.argmax(swag_predictions, axis=1) == targets)
swag_nll = -np.mean(
    np.log(swag_predictions[np.arange(swag_predictions.shape[0]), targets] + eps)
)

eval_ece = losses.ece_score_np(swag_predictions, targets, n_bins=20)
swag_entropies = -np.sum(np.log(swag_predictions + eps) * swag_predictions, axis=1)