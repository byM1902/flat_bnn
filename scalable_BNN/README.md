# Experiment for Scalable BNN
https://arxiv.org/abs/1612.01474


# Baseline
```bash
python3 scalable_BNN/train.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --use_test \
      --dir=log_models/cifar10/scaleBNN/PreResNet164 --mix_sgd --cosine_schedule --max_num_models 5
```
```bash
python3 scalable_BNN/train.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --use_test \
      --dir=log_models/cifar100/scaleBNN/PreResNet164 --mix_sgd --cosine_schedule --max_num_models 5
```

```bash
python3 scalable_BNN/train.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --use_test \
      --dir=log_models/cifar10/scaleBNN/WideResNet28x10 --mix_sgd --cosine_schedule --max_num_models 5
```

```bash
python3 scalable_BNN/train.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --use_test \
      --dir=log_models/cifar100/scaleBNN/WideResNet28x10 --mix_sgd --cosine_schedule --max_num_models 5
```

# Baseline with SAM
```bash
python3 scalable_BNN/train.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --use_test --sam \
      --dir=log_models/cifar10/scaleBNN/PreResNet164_sam --mix_sgd --cosine_schedule --rho 0.05 --max_num_models 3
```

```bash
python3 scalable_BNN/train.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --use_test  --sam \
      --dir=log_models/cifar100/scaleBNN/PreResNet164_sam --rho 0.1 --mix_sgd --cosine_schedule --max_num_models 3
```

```bash
python3 scalable_BNN/train.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --use_test  --sam  --rho 0.05 \
      --dir=log_models/cifar10/scaleBNN/WideResNet28x10_sam --mix_sgd --cosine_schedule --max_num_models 3
```

```bash
python3 scalable_BNN/train.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --use_test  --sam  --rho 0.1  \
      --dir=log_models/cifar100/scaleBNN/WideResNet28x10_sam --mix_sgd --cosine_schedule --max_num_models 3
```


## Eval model

CUDA_VISIBLE_DEVICES=0 python3 scalable_BNN/test_scalable.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=CIFAR10  \ 
--model=PreResNet164 --max_num_models 3 --use_test --dir ../checkpoint_superpod/checkpoint_PreResNet164_cifar10

CUDA_VISIBLE_DEVICES=0 python3 scalable_BNN/test_scalable.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=CIFAR10  \ 
--model=PreResNet164 --max_num_models 3 --use_test --dir ../checkpoint_superpod/checkpoint_PreResNet164_cifar10_sam