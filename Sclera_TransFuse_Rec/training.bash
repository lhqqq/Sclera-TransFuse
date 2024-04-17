CUDA_VISIBLE_DEVICES=2 nohup python -u train_recognition2.py
    --train_root_path=""\
		--train_list="" \
		--save_path="" \
		--model="cross" \
		--config "./configs/architecture.yaml"\
		> train1.log 2>&1 &
