export CUDA_VISIBLE_DEVICES=0
log_path=$1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting/Solar" ]; then
    mkdir ./logs/LongForecasting/Solar
fi
seq_len=96
model_name=Aamba

root_path_name=./dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=solar_AL
data_name=Solar
experiment_date=20241210
random_seed=2021
embed_size=6
K=6
#for pred_len in 96 192 336 720
for pred_len in 96
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --enc_in 137 \
      --dec_in 137 \
      --c_out 137 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 512 \
      --d_state 2\
      --itr 1 \
      --patch_len 16 \
      --stride 8 \
      --individual 0 \
      --train_epochs 10\
      --K $K\
      --embed_size $embed_size\
      --dropout 0.2\
      --itr 1 --batch_size 16 --learning_rate 0.0001 >$log_path/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$experiment_date'K_'$K'embedsitze_'$embed_size.log
done