export CUDA_VISIBLE_DEVICES=0
log_path=$1
model_name=Aamba
embed_size=64
experiment_date=20241219
model_id_name=traffic
seq_len=96
pred_len=96

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --embed_size 64 \
  --itr 1  >$log_path/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$experiment_date'_embed'$embed_size.log
