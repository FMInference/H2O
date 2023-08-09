# H2O 0.05 + 0.05
python run_xsum_h2o.py --model opt-6.7b --output h2o_6.7b_0.05hh.res --hh_ratio 0.05 --gbs 1 --num_gb 1 --percent 100 0 100 0 100 0
# python run_xsum_h2o.py --model opt-30b --output h2o_30b_0.05hh.res --hh_ratio 0.05 --gbs 3 --num_gb 35 --percent 20 80 0 100 0 100
python ../../flexgen/flex_opt.py --model facebook/opt-30b --percent 20 80 0 100 0 100 --gpu-batch-size 3 --num-gpu-batches 35 --cpu --prompt-len 1978 --gen-len 64 --debug fewer_batch --hh-ratio 0.05 --hh-all

# H2O 0.1 + 0.1
python run_xsum_h2o.py --model opt-6.7b --output h2o_6.7b_0.1hh.res --hh_ratio 0.1 --gbs 1 --num_gb 1 --percent 100 0 100 0 100 0
# python run_xsum_h2o.py --model opt-30b --output h2o_30b_0.1hh.res --hh_ratio 0.1 --gbs 3 --num_gb 15 --percent 20 80 0 100 0 100
python ../../flexgen/flex_opt.py --model facebook/opt-30b --percent 20 80 0 100 0 100 --gpu-batch-size 3 --num-gpu-batches 15 --cpu --prompt-len 1978 --gen-len 64 --debug fewer_batch --hh-ratio 0.1 --hh-all

# H2O 0.2 + 0.2
python run_xsum_h2o.py --model opt-6.7b --output h2o_6.7b_0.2hh.res --hh_ratio 0.2 --gbs 1 --num_gb 1 --percent 80 20 100 0 100 0
# python run_xsum_h2o.py --model opt-30b --output h2o_30b_0.2hh.res --hh_ratio 0.2 --gbs 3 --num_gb 10 --percent 20 80 0 100 0 100
python ../../flexgen/flex_opt.py --model facebook/opt-30b --percent 20 80 0 100 0 100 --gpu-batch-size 3 --num-gpu-batches 10 --cpu --prompt-len 1978 --gen-len 64 --debug fewer_batch --hh-ratio 0.2 --hh-all

# H2O 0.3 + 0.3
python run_xsum_h2o.py --model opt-6.7b --output h2o_6.7b_0.3hh.res --hh_ratio 0.3 --gbs 1 --num_gb 1 --percent 80 20 100 0 100 0
# python run_xsum_h2o.py --model opt-30b --output h2o_30b_0.3hh.res --hh_ratio 0.3 --gbs 3 --num_gb 7 --percent 20 80 0 100 0 100
python ../../flexgen/flex_opt.py --model facebook/opt-30b --percent 20 80 0 100 0 100 --gpu-batch-size 3 --num-gpu-batches 7 --cpu --prompt-len 1978 --gen-len 64 --debug fewer_batch --hh-ratio 0.3 --hh-all

# FlexGen (build original FlexGen rather than h2o_flexgen)
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

# HuggingFace example
python3 hf_opt.py --model facebook/opt-1.3b --batch-size 16 --prompt-len 512 --gen-len 64
python3 hf_opt.py --model facebook/opt-30b --batch-size 1 --prompt-len 512 --gen-len 64 --cpu-offload
# DeepSpeed example
deepspeed --num_gpus 1 hf_opt.py --model facebook/opt-1.3b --batch-size 16 --prompt-len 512 --gen-len 64
deepspeed --num_gpus 1 hf_opt.py --model facebook/opt-30b --batch-size 16 --prompt-len 512 --gen-len 64 --cpu-offload
