# model_name=$1
# dataset $2
# checkpoint path $3
#CUDA index $4

## Dropout
CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data  --dataset=$2 --model=$1Drop \
      --use_test --method=Dropout  --file=$3/$1Drop_sam/checkpoint-300.pt --run_name='DropOut' --log_path=$3/$1Drop_sam

#SWA-Dropout:
CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1Drop \
      --cov_mat --use_test --method=SWAGDrop --scale=0. --use_diag --file=$3/$1Drop_sam/swag-300.pt --run_name='SWA_Dropout' --log_path=$3/$1Drop_sam

######### VGG

## Dropout
#CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data  --dataset=$2 --model=$1Drop \
#      --use_test --method=Dropout  --file=$3/$1Drop_sam_rho5e-3/checkpoint-300.pt --run_name='DropOut' --log_path=$3/$1Drop_sam_rho5e-3
#
##SWA-Dropout:
#CUDA_VISIBLE_DEVICES=$4 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=$2 --model=$1Drop \
#      --cov_mat --use_test --method=SWAGDrop --scale=0. --use_diag --file=$3/$1Drop_sam_rho5e-3/swag-300.pt --run_name='SWA_Dropout' --log_path=$3/$1Drop_sam_rho5e-3
