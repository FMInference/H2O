# TODO: this is just a template
# opt-6.7b
time python3 helm_run.py --description summarization_xsum_sampled:model=text,temperature=0.3,device=cpu \
                         --model facebook/opt-6.7b\
                         --percent 0 100 0 100 0 100 \
                         --gpu-batch-size 8 --num-gpu-batches 4 --cpu \
                         --max-eval-instance 518
# opt-30b
time python3 helm_run.py --description summarization_xsum_sampled:model=text,temperature=0.3,device=cpu \
                         --model facebook/opt-30b \
                         --percent 0 100 0 100 0 100 \
                         --gpu-batch-size 8 --num-gpu-batches 4 --cpu \
                         --max-eval-instance 518
