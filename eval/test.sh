# PATH=/root/miniconda2/envs/pytorch/bin:$PATH
cd .. && \
python pseudo_content/model_test.py \
--margin 0.5 \
--caption_model topdown \
--id eval_pch \
--model_pth best \
--input_json data/cocotalk.json \
--input_fc_dir data/cocobu_fc \
--input_att_dir data/cocobu_att \
--input_label_h5 data/cocotalk_label.h5 \
--batch_size 1000 \
--checkpoint_path saving/pch \
--save_checkpoint_every 60 \
--losses_log_every 1 \
--val_images_use -1 \
--max_epochs 30 \
--rnn_size 512 \
--language_eval 1 \
--GPU  2 \
--gen_lr 5e-6 \
--dis_lr 5e-6 \
--lambda_G 1 \
--lambda_obj 1 \
--lambda_rec 1 \
--lambda_tpt  0.1 \
--learning_rate_decay_start -1 \
--learning_rate_decay_rate 0.8 \
--range_netD 1 \
--start_from saving/model-best.pth \
--beam_size 5
