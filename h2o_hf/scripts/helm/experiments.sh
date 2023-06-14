bash scripts/helm/full_cache.sh xsum facebook/opt-30b opt
bash scripts/helm/h2o.sh xsum facebook/opt-30b opt
bash scripts/helm/local.sh xsum facebook/opt-30b opt

bash scripts/helm/full_cache.sh xsum huggyllama/llama-7b llama
bash scripts/helm/h2o.sh xsum huggyllama/llama-7b llama
bash scripts/helm/local.sh xsum huggyllama/llama-7b llama

bash scripts/helm/full_cache.sh xsum EleutherAI/gpt-neox-20b gpt_neox llama
bash scripts/helm/h2o.sh xsum EleutherAI/gpt-neox-20b gpt_neox llama
bash scripts/helm/local.sh xsum EleutherAI/gpt-neox-20b gpt_neox llama





