## model_name=$1
 ## dataset $2
 ## checkpoint path $3
 ##CUDA index $4
 ##seed $5
# CIFAR 100 - seed 0
bash eval_uncertain_seed.sh "VGG16" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "2" "0"
bash eval_uncertain_seed.sh "WideResNet28x10" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "2" "0"
bash eval_uncertain_seed.sh "PreResNet164" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "2" "0"

# CIFAR 100 - seed 2
bash eval_uncertain_seed.sh "VGG16" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "2" "2"
bash eval_uncertain_seed.sh "WideResNet28x10" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "2" "2"
bash eval_uncertain_seed.sh "PreResNet164" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "2" "2"


# CIFAR 10 - seed 0
bash eval_uncertain_seed.sh "VGG16" "CIFAR10" "/vinai/trunglm12/log_models_multi_seed/cifar10/swag_mix" "2" "0"
bash eval_uncertain_seed.sh "WideResNet28x10" "CIFAR10" "/vinai/trunglm12/log_models_multi_seed/cifar10/swag_mix" "2" "0"
bash eval_uncertain_seed.sh "PreResNet164" "CIFAR10" "/vinai/trunglm12/log_models_multi_seed/cifar10/swag_mix" "2" "0"

# CIFAR 10 - seed 2
bash eval_uncertain_seed.sh "VGG16" "CIFAR10" "/vinai/trunglm12/log_models_multi_seed/cifar10/swag_mix" "2" "2"
bash eval_uncertain_seed.sh "WideResNet28x10" "CIFAR10" "/vinai/trunglm12/log_models_multi_seed/cifar10/swag_mix" "2" "2"
bash eval_uncertain_seed.sh "PreResNet164" "CIFAR10" "/vinai/trunglm12/log_models_multi_seed/cifar10/swag_mix" "2" "2"