CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=100 --dataset=CIFAR10 --save_freq=300 \
      --model=VGG16 --lr_init=0.01 --wd=5e-4 --swa --swa_start=61 --swa_lr=0.005 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag_mix/VGG16_sam_rho1e-2_lr0.01 --rho 1e-2 --mix_sgd --cosine_schedule   -> acc 92.6900

CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=100 --dataset=CIFAR10 --save_freq=300 \
      --model=VGG16 --lr_init=0.01 --wd=5e-4 --swa --swa_start=61 --swa_lr=0.005 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag_mix/VGG16_sam_rho5e-3 --rho 5e-3 --mix_sgd --cosine_schedule       -> acc 92.870

CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=100 --dataset=CIFAR10 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --swa --swa_start=61 --swa_lr=0.005 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag_mix/VGG16_sam_rho5e-4 --rho 5e-4 --mix_sgd --cosine_schedule         -> acc 92.340

CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=100 --dataset=CIFAR10 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --swa --swa_start=61 --swa_lr=0.005 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag_mix/VGG16_sam_rho5e-5 --rho 5e-5 --mix_sgd --cosine_schedule         -> acc 92.510