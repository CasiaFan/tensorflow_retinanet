export PYTHONPATH="$PYTHONPATH:/home/arkenstone/models/research:/data/fanzong/models/research/slim"

python3 model_main.py \
--model_dir="train" --pipeline_config_path="retinanet_50_train.config" 