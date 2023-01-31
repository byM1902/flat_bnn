## Flat Seeking Bayesian Neural Networks
This is the official implementation of **F-BNN: Flat Seeking Bayesian Neural Networks**
### Install envs
USing Annaconda to install  
`conda env create -f swag_envs.yml`

### Dataset
Create a folder that includes `cifar10`, `cifar100` and `imagenet` folder for these three dataset  
For the experiments of Imagenet, please refer folder   `experiments/imagenet`, the instruction below is for experiments with CIFAR10 and CIFAR100


### Training model
#### CIFAR10

[//]: # (CUDA_VISIBLE_DEVICES=1 python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR10 --save_freq=300 --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --dir=log_models/cifar10/swag/PreResNet164)

##### PreResNet164 / PreResNet164Drop
- swag_mix_drop
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=PreResNet164Drop --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag_mix/PreResNet164Drop_sam --mix_sgd --cosine_schedule --rho 0.05
```
- swag_mix
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag_mix/PreResNet164_sam_seed234 --mix_sgd --cosine_schedule --rho 0.05 --seed 234
```
- sgd  
```bash
python experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --use_test --sam --dir=log_models/cifar10/sgd/PreResNet164_sam --rho 0.05
```
- swag  
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --sam --rho 0.05 \
      --dir=log_models/cifar10/swag/PreResNet164_sam
```

##### WideResNet28x10 / WideResNet28x10Drop
- swag_mix_drop  
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=WideResNet28x10Drop --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --sam --rho 0.05 \
      --dir=log_models/cifar10/swag_mix/WideResNet28x10Drop_sam --mix_sgd --cosine_schedule
```

- swag_mix  
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --sam --rho 0.05 \
      --dir=log_models/cifar10/swag_mix/WideResNet28x10_sam --mix_sgd --cosine_schedule
```

- sgd  
```bash
python experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --use_test --sam --dir=log_models/cifar10/sgd/WideResNet28x10_sam --rho 0.05
```
- swag  
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR10 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --sam \
      --dir=log_models/cifar10/swag/WideResNet28x10_sam --rho 0.05
```

#### CIFAR100
##### PreResNet164
- swag_mix_drop  
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164Drop --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/PreResNet164Drop_sam --rho 0.1 --mix_sgd --cosine_schedule
```
- swag_mix_drop_baseline  
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164Drop --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/PreResNet164Drop_sam --rho 0.1 --mix_sgd --cosine_schedule
```
- swag_mix  
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/PreResNet164_sam --rho 0.1 --mix_sgd --cosine_schedule
```
- swag_mix_baseline  
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test \
      --dir=log_models/cifar100/swag_mix/PreResNet164 --mix_sgd --cosine_schedule
```

- sgd  
```bash
python experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --use_test  --sam --dir=log_models/cifar100/sgd/PreResNet164_sam  --rho 0.1
```
- swag  
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag/PreResNet164_sam --rho 0.1
```
##### WideResNet28x10

- swag_mix_drop  
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=WideResNet28x10Drop --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/WideResNet28x10Drop_sam --rho 0.1 --mix_sgd --cosine_schedule
  ```
- swag_mix  
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag_mix/WideResNet28x10_sam --rho 0.1 --mix_sgd --cosine_schedule
```
- sgd  
```bash
python experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --use_test  --sam --dir=log_models/cifar100/sgd/WideResNet28x10_sam --rho 0.1
```
- swag  
```bash
python3 experiments/train/run_swag.py --data_path=./datasets --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test  --sam \
      --dir=log_models/cifar100/swag/WideResNet28x10_sam --rho 0.1
```
    
### Evaluate the uncertainty of model

- $1: model: {PreResNet164, WideResNet28x10}
- $2: dataset: {CIFAR10, CIFAR100}
- $3: log-path: {log_models/cifar10/swag_mix, log_models/cifar100/swag_mix}
- $4: Cuda idx
After eval, please check file `<log-path>/uncertainty_eval.txt`
#### Example eval on CIFAR100

`bash script/eval_uncertain.sh "WideResNet28x10" "CIFAR100" "log_models/cifar100/swag_mix" "1"`
`bash script/eval_uncertain.sh "PreResNet164" "CIFAR100" "log_models/cifar100/swag_mix" "1"`


#### Example eval on CIFAR10
`bash script/eval_uncertain.sh "WideResNet28x10" "CIFAR10" "log_models/cifar10/swag_mix" "1"`
`bash script/eval_uncertain.sh "PreResNet164" "CIFAR10" "log_models/cifar10/swag_mix" "1"`
