# model_name=$1
# dataset $2
# checkpoint path $3
#CUDA index $4

# SGD
CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1 --use_test \
      --method=SGD --N=1 --file=$3/$1_sam/checkpoint-300.pt --log_path=$3/$1_sam --run_name='SGD'

# SWA:
CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1 --use_test \
      --cov_mat --method=SWAG --use_diag --N=1 --scale=0. --file=$3/$1_sam/swag-300.pt  --log_path=$3/$1_sam --run_name='SWA'

# SWAG-0.5:
CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1 --use_test \
      --cov_mat --method=SWAG --scale=0.5 --file=$3/$1_sam/swag-300.pt  --log_path=$3/$1_sam --run_name='SWAG-scale=0.5'

# SWAG-0.1:
CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1 --use_test \
      --cov_mat --method=SWAG --scale=0.1 --file=$3/$1_sam/swag-300.pt  --log_path=$3/$1_sam --run_name='SWAG-scale=0.1'

# SWAG-Diagonal-1:
CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1 --use_test \
      --cov_mat --method=SWAG --use_diag --file=$3/$1_sam/swag-300.pt --log_path=$3/$1_sam --run_name='SWAG-Diagonal-scale=1'

# SWAG-Diagonal-0.2:
CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1 --use_test \
      --cov_mat --method=SWAG --use_diag --scale=0.2 --file=$3/$1_sam/swag-300.pt --log_path=$3/$1_sam --run_name='SWAG-Diagonal-scale=0.2'


################## VGG16
# model_name=$1
# dataset $2
# checkpoint path $3
#CUDA index $4

## SGD
#CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1 --use_test \
#      --method=SGD --N=1 --file=$3/$1_sam_rho5e-3/checkpoint-300.pt --log_path=$3/$1_sam_rho5e-3 --run_name='SGD'
#
## SWA:
#CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1 --use_test \
#      --cov_mat --method=SWAG --use_diag --N=1 --scale=0. --file=$3/$1_sam_rho5e-3/swag-300.pt  --log_path=$3/$1_sam_rho5e-3 --run_name='SWA'
#
## SWAG-0.5:
#CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1 --use_test \
#      --cov_mat --method=SWAG --scale=0.5 --file=$3/$1_sam_rho5e-3/swag-300.pt  --log_path=$3/$1_sam_rho5e-3 --run_name='SWAG-scale=0.5'
#
## SWAG-0.1:
#CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1 --use_test \
#      --cov_mat --method=SWAG --scale=0.1 --file=$3/$1_sam_rho5e-3/swag-300.pt  --log_path=$3/$1_sam_rho5e-3 --run_name='SWAG-scale=0.1'
#
## SWAG-Diagonal-1:
#CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1 --use_test \
#      --cov_mat --method=SWAG --use_diag --file=$3/$1_sam_rho5e-3/swag-300.pt --log_path=$3/$1_sam_rho5e-3 --run_name='SWAG-Diagonal-scale=1'
#
## SWAG-Diagonal-0.2:
#CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1 --use_test \
#      --cov_mat --method=SWAG --use_diag --scale=0.2 --file=$3/$1_sam_rho5e-3/swag-300.pt --log_path=$3/$1_sam_rho5e-3 --run_name='SWAG-Diagonal-scale=0.2'
