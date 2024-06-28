export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

/opt/software/install/miniconda37/bin/python train.py \
    --model_name_or_path llama/7b-32 \
    --data_path data/train_data_points-v2.1-64.json \
    --bf16 True \
    --output_dir out \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --full_determinism \
    --tf32 False \
    --report_to tensorboard

