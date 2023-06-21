# H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

## Installation

**Requirements**

- PyTorch >= 1.12

```
pip install crfm-helm
pip install git+https://github.com/huggingface/transformers
pip install lm-eval
```


```
import helm
import os, shutil
install_path = helm.__file__
source_path = 'helm/src/helm/benchmark/metrics/toxicity_metrics.py'
target_path = '/'.join(install_path.split('/')[:-1])
target_path = os.path.join(target_path, 'benchmark/metrics/toxicity_metrics.py')
shutil.copy(source_path, target_path) # modify toxicity_metrics.py
```



## Usage and Examples

### Text-Generation with custom prompts

To get started, you can generate text with your own prompts, The models will automatically downloaded from Hugging Face.

```
python -u run_text_generation.py \
    --model_arch llama \
    --model_name huggyllama/llama-13b \
    --recent_ratio 0.1 \ # kv cache size for the most recent ones (num = 0.1 * length_of_prompt)
    --heavy_ratio 0.1 \ # kv cache size heavy hitters (num = 0.1 * length_of_prompt)
```

You can change the prompt by modifying **prompt_text** in **run_test_generation.py**, more examples are available in **scripts/generation**.

### Evaluation on tasks from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework

Here we provide an example to evaluate the 5-shot performance of LLaMA-7b on OpenbookQA, more examples can be found at **scripts/lm_eval/experiments.sh**

```
# Step 1: Prepare inference text
task=openbookqa
shots=5
python -u generate_task_data.py \
  --output-file ${task}-${shots}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots}

# Step 2 (Full Cache Baseline): Generate the output from LLaMA-7b with Full Cache
model=huggyllama/llama-7b
model_arch=llama
python -u run_lm_eval_harness.py \
  --input-path ${task}-${shots}.jsonl \
  --output-path ${task}-${shots}-${model_arch}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch}
  
# Step 2 ("Local" Baseline): Generate the output from LLaMA-7b with 20% kv of the most recent tokens
model=huggyllama/llama-7b
model_arch=llama
python -u run_lm_eval_harness.py \
  --input-path ${task}-${shots}.jsonl \
  --output-path ${task}-${shots}-${model_arch}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch} \
  --enable_small_cache \
  --heavy_ratio 0 \
  --recent_ratio 0.2

# Step 2 (H2O): Generate the output from LLaMA-7b with H2O
model=huggyllama/llama-7b
model_arch=llama
python -u run_lm_eval_harness.py \
  --input-path ${task}-${shots}.jsonl \
  --output-path ${task}-${shots}-${model_arch}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch} \
  --enable_small_cache \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1

# Step 3: Evaluate the performance of generated text
python -u evaluate_task_result.py \
  --result-file ${task}-${shots}-${model_arch}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --model-type ${model_arch}
```

### Evaluation on tasks from [HELM](https://crfm.stanford.edu/helm/latest/) framework

To evaluate the performance of tasks from HELM framework, the pipeline is similar with lm-eval-harness. An example is provided in the following, and more experiments can be found at **scripts/helm/experiments.sh**

```
# Step 1: prepare inference text
# Examples of converting inference data to jsonl format is provided in helm/command/get_data.sh
# And the data is provided in data/

# Step 2 (Full Cache Baseline): Generate the output from LLaMA-7b with Full Cache
model=huggyllama/llama-7b
model_arch=llama
python -u run_helm.py \
  --input_path data/xsum.jsonl \
  --output_path generate_xsum_llama7b.jsonl \
  --model_name ${model} \
  --model_arch ${model_arch} 

# Step 2 ("Local" Baseline): Generate the output from LLaMA-7b with 20% kv of the most recent tokens
model=huggyllama/llama-7b
model_arch=llama
python -u run_helm.py \
  --input_path data/xsum.jsonl \
  --output_path generate_xsum_llama7b_local.jsonl \
  --model_name ${model} \
  --model_arch ${model_arch} \
  --enable_small_cache \
  --heavy_ratio 0 \
  --recent_ratio 0.2
  
# Step 2 (H2O): Generate the output from LLaMA-7b with H2O
model=huggyllama/llama-7b
model_arch=llama
python -u run_helm.py \
  --input_path data/xsum.jsonl \
  --output_path generate_xsum_llama7b_h20.jsonl \
  --model_name ${model} \
  --model_arch ${model_arch} \
  --enable_small_cache \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1
  
# Step 3: Evaluate the performance of generated text (refer helm/command/eval.sh)
cd helm
TASK=xsum
JSONL=generate_xsum_llama7b.jsonl
OUTPUT=xsum_llama7b_result
ARCH=llama
python scripts/offline_eval/import_results.py together ${JSONL} --cache-dir prod_env/cache
helm-run --conf src/helm/benchmark/presentation/${TASK}/run_specs_${ARCH}.conf --local --max-eval-instances 100 --num-train-trials=1 --suite ${OUTPUT} -n 1
helm-summarize --suite ${OUTPUT} 
# The results are writted into a tex file that can be found in benchmark_output/runs/xsum_llama7b_result/groups/latex/ 
```

