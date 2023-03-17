DEVICE=0
DATA_SET='weibo'
MODEL_CLASS='lebert-softmax'
LR=1e-5
CRF_LR=1e-3
ADAPTER_LR=1e-3
PRETRAIN_MODEL='bert-base-chinese'
#export CUDA_VISIBLE_DEVICES=${DEVICE}
python train.py \
    --device gpu \
    --output_path output/ \
    --add_layer 1 \
    --loss_type ce \
    --lr ${LR} \
    --crf_lr ${CRF_LR} \
    --adapter_lr ${ADAPTER_LR} \
    --weight_decay 0.01 \
    --eps 1.0e-08 \
    --epochs 30 \
    --batch_size_train 32 \
    --batch_size_eval 256 \
    --num_workers 0 \
    --eval_step 100 \
    --max_seq_len 150 \
    --max_word_num  3 \
    --max_scan_num 3000000 \
    --data_path datasets/ner_data/${DATA_SET}/ \
    --dataset_name ${DATA_SET} \
    --model_class ${MODEL_CLASS} \
    --pretrain_model_path pretrain_model/${PRETRAIN_MODEL} \
    --pretrain_embed_path  /root/autodl-tmp/project/pretrain_model/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt\
    --seed 42 \
    --markup bios \
    --grad_acc_step 1 \
    --max_grad_norm 1.0 \
    --num_workers 0 \
    --warmup_proportion 0.1 \
    --load_word_embed \
    --do_train \
    --do_eval