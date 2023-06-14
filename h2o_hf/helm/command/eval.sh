TASK=xsum
JSONL=$1 # Jsonl file for evaluation
OUTPUT=$2 # result-output-direction
ARCH=llama
# arch in [opt, gptneox, llama]
python scripts/offline_eval/import_results.py together ${JSONL} --cache-dir prod_env/cache
helm-run --conf src/helm/benchmark/presentation/${TASK}/run_specs_${ARCH}.conf --local --max-eval-instances 1000 --num-train-trials=1 --suite ${OUTPUT} -n 1
helm-summarize --suite ${OUTPUT}
