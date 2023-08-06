# H2O


# FlexGen
python run_xsum_flexgen.py --model opt-6.7b --output flexgen_6.7b.res
python run_xsum_flexgen.py --model opt-30b --output flexgen_30b.res

# python run_xsum_flexgen.py --sample_num 10 --model opt-6.7b --output flexgen_6.7b.res
# python run_xsum_flexgen.py --sample_num 10 --model opt-30b --output flexgen_30b.res

# do not use: this is just a template for helm_run.py
# opt-6.7b
# time python3 helm_run.py --description summarization_xsum_sampled:model=text,temperature=0.3,device=cpu \
#                          --model facebook/opt-6.7b\
#                          --percent 0 100 0 100 0 100 \
#                          --gpu-batch-size 8 --num-gpu-batches 4 --cpu \
#                          --max-eval-instance 518
# opt-30b
# time python3 helm_run.py --description summarization_xsum_sampled:model=text,temperature=0.3,device=cpu \
#                          --model facebook/opt-30b \
#                          --percent 0 100 0 100 0 100 \
#                          --gpu-batch-size 8 --num-gpu-batches 4 --cpu \
#                          --max-eval-instance 518
