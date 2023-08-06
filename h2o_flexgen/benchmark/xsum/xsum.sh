# H2O 0.1 + 0.1
python run_xsum_h2o.py --model opt-6.7b --output h2o_6.7b_0.1hh.res --hh_ratio 0.1 --gbs 1 --num_gb 1 --percent 100 0 100 0 100 0
python run_xsum_h2o.py --model opt-30b --output h2o_30b_0.1hh.res --hh_ratio 0.1 --gbs 3 --num_gb 15 --percent 20 80 0 100 0 100
# python flexgen/flex_opt.py --model facebook/opt-30b --percent 20 80 0 100 0 100 --gpu-batch-size 3 --num-gpu-batches 15 --cpu --prompt-len 1978 --gen-len 64 --debug fewer_batch --hh-ratio 0.1 --hh-all

# H2O 0.2 + 0.2
python run_xsum_h2o.py --model opt-6.7b --output h2o_6.7b_0.2hh.res --hh_ratio 0.2 --gbs 1 --num_gb 1 --percent 80 20 100 0 100 0
python run_xsum_h2o.py --model opt-30b --output h2o_30b_0.2hh.res --hh_ratio 0.2 --gbs 3 --num_gb 10 --percent 20 80 0 100 0 100
# python flexgen/flex_opt.py --model facebook/opt-30b --percent 20 80 0 100 0 100 --gpu-batch-size 3 --num-gpu-batches 10 --cpu --prompt-len 1978 --gen-len 64 --debug fewer_batch --hh-ratio 0.2 --hh-all

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
