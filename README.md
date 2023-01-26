# Flat Seeking Bayesian Neural Networks

## SWAG + SAM
### Install envs
    USing Annaconda to install
`conda env create -f swag_envs.yml`

### Dataset
Create a folder that includes `cifar10` and `cifar100` folder for these two dataset

### Training model
#### CIFAR10

[//]: # (CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --dir=log_models/cifar10/swag/PreResNet164)

##### VGG16 / VGG16Drop

- TUNNING swag_mix: rho = {5e-4, 5e-5, 5e-6}
```bash
CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=100 --dataset=CIFAR10 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --swa --swa_start=61 --swa_lr=0.005 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag_mix/VGG16_sam_rho5e-4 --rho 5e-4 --mix_sgd --cosine_schedule
```

- swag_mix_drop
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=VGG16Drop --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.005 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag_mix/VGG16Drop_sam  --rho 5e-3 --mix_sgd --cosine_schedule
```
- swag_mix
```bash
CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.005 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag_mix/VGG16_sam --rho 5e-3 --mix_sgd --cosine_schedule
```
- sgd
```bash
python experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --use_test --sam --dir=log_models/cifar10/sgd/VGG16_sam --sam_adaptive
```
- swag
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.005 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag/VGG16_sam  --sam_adaptive
```
##### PreResNet164 / PreResNet164Drop
- swag_mix_drop
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=PreResNet164Drop --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag_mix/PreResNet164Drop_sam --mix_sgd --cosine_schedule --rho 0.05
```
- swag_mix
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag_mix/PreResNet164_sam_seed234 --mix_sgd --cosine_schedule --rho 0.05 --seed 234
```
- sgd  
```bash
python experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --use_test --sam --dir=log_models/cifar10/sgd/PreResNet164_sam --rho 0.05
```
- swag  
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --sam --rho 0.05 \
      --dir=log_models/cifar10/swag/PreResNet164_sam
```

##### WideResNet28x10 / WideResNet28x10Drop
- swag_mix_drop  
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=WideResNet28x10Drop --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --sam --rho 0.05 \
      --dir=log_models/cifar10/swag_mix/WideResNet28x10Drop_sam --mix_sgd --cosine_schedule
```

- swag_mix  
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --sam --rho 0.05 \
      --dir=log_models/cifar10/swag_mix/WideResNet28x10_sam --mix_sgd --cosine_schedule
```

- sgd  
```bash
python experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --use_test --sam --dir=log_models/cifar10/sgd/WideResNet28x10_sam --rho 0.05
```
- swag  
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag/WideResNet28x10_sam --rho 0.05
```

#### CIFAR100
##### VGG-16 / VGG16Drop
- swag_mix_drop  
```bash
CUDA_VISIBLE_DEVICES=5 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=VGG16Drop --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/VGG16Drop_sam --rho 5e-3 --mix_sgd --cosine_schedule
```
- swag_mix  
```bash
CUDA_VISIBLE_DEVICES=5 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/VGG16_sam --rho 5e-3 --mix_sgd --cosine_schedule
```
- sgd
```bash
python experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --use_test  --sam --dir=log_models/cifar100/sgd/VGG16_sam --sam_adaptive --rho 1
```
- swag  
```bash
CUDA_VISIBLE_DEVICES=5 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag/VGG16_sam --sam_adaptive --rho 1
```
##### PreResNet164
- swag_mix_drop  
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164Drop --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/PreResNet164Drop_sam --rho 0.1 --mix_sgd --cosine_schedule
```
- swag_mix_drop_baseline  
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164Drop --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/PreResNet164Drop_sam --rho 0.1 --mix_sgd --cosine_schedule
```
- swag_mix  
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/PreResNet164_sam --rho 0.1 --mix_sgd --cosine_schedule
      
CUDA_VISIBLE_DEVICES=0,1 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=100 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=0 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/PreResNet164_sam_sharpness_300e --rho 0.1 --mix_sgd --cosine_schedule --eval_sharpness

CUDA_VISIBLE_DEVICES=0 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=100 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=0 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/PreResNet164_sharpness_300e --rho 0.1 --mix_sgd --cosine_schedule --use_sgd --eval_sharpness
      
CUDA_VISIBLE_DEVICES=6 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=100 --dataset=CIFAR100 --save_freq=100 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=0 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/PreResNet164_sam_sharpness_100e --rho 0.1 --mix_sgd --cosine_schedule --eval_sharpness

CUDA_VISIBLE_DEVICES=0 python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=100 --dataset=CIFAR100 --save_freq=100 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=0 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/PreResNet164_sharpness_100e --rho 0.1 --mix_sgd --cosine_schedule --use_sgd --eval_sharpness
```
- swag_mix_baseline  
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test \
      --dir=log_models/cifar100/swag_mix/PreResNet164 --mix_sgd --cosine_schedule
```
log_models/cifar100/swag_mix/PreResNet164/swag-300.pt
- sgd  
```bash
python experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --use_test  --sam --dir=log_models/cifar100/sgd/PreResNet164_sam  --rho 0.1
```
- swag  
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag/PreResNet164_sam --rho 0.1
```
##### WideResNet28x10

- swag_mix_drop  
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=WideResNet28x10Drop --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/WideResNet28x10Drop_sam --rho 0.1 --mix_sgd --cosine_schedule
  ```
- swag_mix  
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/WideResNet28x10_sam --rho 0.1 --mix_sgd --cosine_schedule
```
- sgd  
```bash
python experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --use_test  --sam --dir=log_models/cifar100/sgd/WideResNet28x10_sam --rho 0.1
```
- swag  
```bash
python3 experiments/train/run_swag.py --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag/WideResNet28x10_sam --rho 0.1
```
    
### Evaluate the uncertainty of model

- $1: model: {VGG16, PreResNet164, WideResNet28x10}
- $2: dataset: {CIFAR10, CIFAR100}
- $3: log-path: {log_models/cifar10/swag_mix, log_models/cifar100/swag_mix}
- $4: Cuda idx
After eval, please check file `<log-path>/uncertainty_eval.txt`
#### Example eval on CIFAR100

`bash eval_uncertain.sh "WideResNet28x10" "CIFAR100" "log_models/cifar100/swag_mix" "1"`
`bash eval_uncertain.sh "VGG16" "CIFAR100" "log_models/cifar100/swag_mix" "1"`
using VGG16_sam_rho5e-3, check 'log_models/cifar100/swag_mix/VGG16_sam_rho5e-3/uncer..', 'log_models/cifar100/swag_mix/VGG16Drop_sam_rho5e-3/uncer..'
`bash eval_uncertain.sh "PreResNet164" "CIFAR100" "/vinai/trunglm12/swag-sam/cifar100/swag_mix" "1"`


#### Example eval on CIFAR10
`bash eval_uncertain.sh "WideResNet28x10" "CIFAR10" "log_models/cifar10/swag_mix" "1"`
"/vinai/trunglm12/drop_sam/cifar10/swag_mix/WideResNet28x10Drop_sam/uncer"
`bash eval_uncertain.sh "VGG16" "CIFAR10" "log_models/cifar10/swag_mix" "1"`
using VGG16_sam_rho5e-3, check 'log_models/cifar10/swag_mix/VGG16_sam_rho5e-3/uncer..', 'log_models/cifar10/swag_mix/VGG16Drop_sam_rho5e-3/uncer..'
`bash eval_uncertain.sh "PreResNet164" "CIFAR10" "/vinai/trunglm12/swag-sam/cifar10/swag_mix" "1"`


`bash eval_uncertain_seed.sh "VGG16" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "2" "0"`
`bash eval_uncertain_seed.sh "WideResNet28x10" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "2" "0"`
`bash eval_uncertain_seed.sh "PreResNet164" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "2" "0"`


#### Eval baseline
```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=CIFAR100 --model=PreResNet164 --use_test \
      --cov_mat --method=SWAG --use_diag --N=1 --scale=0. --file=/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix/PreResNet164-baseline/swag-300.pt  \
      --log_path=/home/ubuntu/swa_gaussian/log_models/eval_result/CIFAR100_PreResNet164_baseline.txt --run_name='SWA' --seed 1
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=CIFAR100 --model=PreResNet164 --use_test \
      --cov_mat --method=SWAG --scale=0.1 --file=/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix/PreResNet164-baseline/swag-300.pt \
      --log_path=/home/ubuntu/swa_gaussian/log_models/eval_result/CIFAR100_PreResNet164_baseline.txt  --run_name='SWAG-scale=0.1' --seed 1

# SWAG-Diagonal-0.2:
CUDA_VISIBLE_DEVICES=0 python3 experiments/uncertainty/uncertainty.py  --data_path=/home/ubuntu/vit_selfOT/ViT-pytorch/data --dataset=CIFAR100 --model=PreResNet164 --use_test \
      --cov_mat --method=SWAG --use_diag --scale=0.2 --file=/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix/PreResNet164-baseline/swag-300.pt \
      --log_path=/home/ubuntu/swa_gaussian/log_models/eval_result/CIFAR100_PreResNet164_baseline.txt  --run_name='SWAG-Diagonal-scale=0.2' --seed 1
```
