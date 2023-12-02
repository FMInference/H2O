task=$1
shots=$2
method=$3
GPU=$4
HH_SIZE=$5
RECENT_SIZE=$6

if [[ ${method} == 'h2o' ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python -u run_summarization.py \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_h2o_hh${1}_local${2}.jsonl \
        --model_name meta-llama/Llama-2-7b-hf \
        --hh_size ${HH_SIZE} \
        --recent_size ${RECENT_SIZE} \
        --cache_dir ../../llm_weights \
        --enable_h2o_cache
elif [[ ${method} == 'full' ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python -u run_summarization.py \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_full.jsonl \
        --model_name meta-llama/Llama-2-7b-hf \
        --cache_dir ../../llm_weights 
else
    echo 'unknown argment for method'
fi
