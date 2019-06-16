export PYTHONPATH="$PYTHONPATH:/PATH/TO/models/research:/PATH/TO/models/research/slim"

python3 model_main.py \
--model_dir="train" --pipeline_config_path="retinanet_50_train.config" 