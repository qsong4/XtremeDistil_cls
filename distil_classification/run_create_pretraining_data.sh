BERT_BASE_DIR=./chinese_roberta_wwm_ext_L-12_H-768_A-12
python create_pretraining_data.py \
  --input_file=./data/sample_text.txt \
  --output_file=./data/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=64 \
  --max_predictions_per_seq=6 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=3 \
  --do_whole_word_mask=True