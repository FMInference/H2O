max_instances=1000
num_train_trails=1
suite=xsum_opt
specs=opt
suite=xsum_llama
specs=llama
suite=xsum_gptneox
specs=gptneox
helm-run --conf helm/src/helm/benchmark/presentation/xsum/run_specs_$specs.conf --local --max-eval-instances $max_instances --num-train-trials=$num_train_trails --suite $suite -n 1 --dry-run









