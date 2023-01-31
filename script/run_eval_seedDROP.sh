
# CIFAR 100 - seed 0
bash eval_drop_seed.sh "VGG16" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "0" "0"
bash eval_drop_seed.sh "WideResNet28x10" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "0" "0"
bash eval_drop_seed.sh "PreResNet164" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "0" "0"

# CIFAR 100 - seed 2
bash eval_drop_seed.sh "VGG16" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "0" "2"
bash eval_drop_seed.sh "WideResNet28x10" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "0" "2"
bash eval_drop_seed.sh "PreResNet164" "CIFAR100" "/vinai/trunglm12/log_models_multi_seed/cifar100/swag_mix" "0" "2"


# CIFAR 10 - seed 0
bash eval_drop_seed.sh "VGG16" "CIFAR10" "/vinai/trunglm12/log_models_multi_seed/cifar10/swag_mix" "0" "0"
bash eval_drop_seed.sh "WideResNet28x10" "CIFAR10" "/vinai/trunglm12/log_models_multi_seed/cifar10/swag_mix" "0" "0"
bash eval_drop_seed.sh "PreResNet164" "CIFAR10" "/vinai/trunglm12/log_models_multi_seed/cifar10/swag_mix" "0" "0"

# CIFAR 10 - seed 2
bash eval_drop_seed.sh "VGG16" "CIFAR10" "/vinai/trunglm12/log_models_multi_seed/cifar10/swag_mix" "0" "2"
bash eval_drop_seed.sh "WideResNet28x10" "CIFAR10" "/vinai/trunglm12/log_models_multi_seed/cifar10/swag_mix" "0" "2"
bash eval_drop_seed.sh "PreResNet164" "CIFAR10" "/vinai/trunglm12/log_models_multi_seed/cifar10/swag_mix" "0" "2"