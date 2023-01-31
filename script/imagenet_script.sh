CUDA_VISIBLE_DEVICES=1 python3 experiments/imagenet/run_swag_imagenet.py
--data_path=/home/ubuntu/datasets/image-net --epochs=10 --model=densenet161
--pretrained --lr_init=0.001 --swa --swa_start=0 --swa_lr=0.02 --cov_mat --use_test
--dir=log_models/imgnet/swag_mix/densenet161_sam --sam --rho 0.05 --batch_size 256 --mix_sgd --cosine_schedule --save_freq=1


# eval imageNet
CUDA_VISIBLE_DEVICES=1 python3 experiments/imagenet/eval_swag_imagenet.py --data_path=<datapath> \
--batch_size=128 --model=densenet161 --ckpt=log_models/imgnet/swag_mix/densenet161_sam/swag-10.pt \
--ckpt_sgd=log_models/imgnet/swag_mix/densenet161_sam/checkpoint-10.pt --scale 0.1 --cov_mat

CUDA_VISIBLE_DEVICES=1 python3 experiments/imagenet/run_swag_imagenet.py
--data_path=<> --epochs=10 --model=resnet152
--pretrained --lr_init=0.001 --swa --swa_start=0 --swa_lr=0.02 --cov_mat --use_test
--dir=log_models/imgnet/swag_mix/resnet152_sam --sam --rho 0.05 --batch_size 256 --mix_sgd --cosine_schedule --save_freq=1


CUDA_VISIBLE_DEVICES=1 python3 experiments/imagenet/eval_swag_imagenet.py --data_path=<datapath> \
--batch_size=128 --model=resnet152 --ckpt=log_models/imgnet/swag_mix/resnet152_sam/swag-10.pt \
--ckpt_sgd=log_models/imgnet/swag_mix/resnet152_sam/checkpoint-10.pt --scale 0.1 --cov_mat