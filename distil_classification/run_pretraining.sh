BERT_BASE_DIR=./chinese_roberta_wwm_ext_L-12_H-768_A-12
python run_pretraining.py \
  --input_file=./data/tf_examples.tfrecord \
  --output_dir=./pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=64 \
  --max_predictions_per_seq=6 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5