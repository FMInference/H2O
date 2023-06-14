task=$1
model=$2
model_arch=$3

if [[ ${model_arch} == 'opt' ]]; then
    input_data=data/${task}_opt.jsonl
else
    input_data=data/${task}.jsonl
fi

python -u run_helm.py \
    --input_path ${input_data} \
    --output_path ${task}-${model_arch}-local.jsonl \
    --model_name ${model} \
    --model_arch ${model_arch} \
    --enable_small_cache \
    --heavy_ratio 0 \
    --recent_ratio 0.2 

