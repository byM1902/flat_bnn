# note: change data_path to <path> (that include: <path>/cifar10, <path>/cifar100 )

#cifar100
#python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
#      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test \
#      --dir=cifar100/swa/PreResNet164
#### PreResNet164
CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag/PreResNet164_sam

#CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
#      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --use_test  --sam \
#      --dir=log_models/cifar100/swag_diag/PreResNet164_sam

CUDA_VISIBLE_DEVICES=1 python experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --use_test  --sam --dir=log_models/cifar100/sgd/PreResNet164_sam

#### VGG-16

#### uncertainty
#swag
CUDA_VISIBLE_DEVICES=1 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=CIFAR100 --model=PreResNet164 --use_test \
      --cov_mat --method=SWAG --scale=0.5 --file=/home/ubuntu/swa_gaussian/cifar100/swa/PreResNet164_sam/swag-300.pt --save_path=/home/ubuntu/swa_gaussian/cifar100/swa/PreResNet164_sam
#swag-diag
CUDA_VISIBLE_DEVICES=1  python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=CIFAR100 --model=PreResNet164 --use_test \
      --cov_mat --method=SWAG --use_diag --file=/home/ubuntu/swa_gaussian/cifar100/swa/PreResNet164_sam/swag-300.pt --save_path=/home/ubuntu/swa_gaussian/cifar100/swa/PreResNet164_sam
##cifar10

#CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
#      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test \
#      --dir=log_models/cifar10/swag/PreResNet164

#### PreResNet164
CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag/PreResNet164_sam

#CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
#      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --use_test --sam \
#      --dir=log_models/cifar10/swag_diag/PreResNet164_sam

CUDA_VISIBLE_DEVICES=7 python experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --use_test --sam --dir=log_models/cifar10/sgd/PreResNet164_sam


###uncertainty
#sgd
CUDA_VISIBLE_DEVICES=2 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data--dataset=CIFAR10 --model=PreResNet164 --use_test \
      --method=SGD --N=1 --file=/home/ubuntu/swa_gaussian/cifar10/sgd/PreResNet164_sam/checkpoint-300.pt --save_path=/home/ubuntu/swa_gaussian/cifar10/sgd/PreResNet164_sam

#swag
CUDA_VISIBLE_DEVICES=2 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=CIFAR10 --model=PreResNet164 --use_test \
      --cov_mat --method=SWAG --scale=0.5 --file=/home/ubuntu/swa_gaussian/cifar10/swa/PreResNet164_sam/swag-300.pt --save_path=/home/ubuntu/swa_gaussian/cifar10/swa/PreResNet164_sam
#swag-diag
CUDA_VISIBLE_DEVICES=2  python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=CIFAR10 --model=PreResNet164 --use_test \
      --cov_mat --method=SWAG --use_diag --file=/home/ubuntu/swa_gaussian/cifar10/swa/PreResNet164_sam/swag-300.pt --save_path=/home/ubuntu/swa_gaussian/cifar10/swa/PreResNet164_sam
#### VGG-16

CUDA_VISIBLE_DEVICES=2 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=CIFAR100 --model=PreResNet164 --use_test \
      --method=SGD --N=1 --file=/vinai/trunglm12/log_models/cifar100/sgd/PreResNet164_sam/checkpoint-300.pt --save_path=/vinai/trunglm12/log_models/cifar100/sgd/PreResNet164_sam

CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300   --model=VGG16 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test  --sam    --dir=log_models/cifar100/swag/VGG16_sam_adaptive_mix_cosLR --sam_adaptive --rho 1 --cosine_schedule