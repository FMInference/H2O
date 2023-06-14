task=$1
shots=5
python -u generate_task_data.py --output-file ${task}-${shots}.jsonl --task-name ${task} --num-fewshot ${shots}
