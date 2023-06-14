import argparse
import json
import os

from lm_eval import evaluator, tasks
from tasks import EvalHarnessAdaptor

def json_to_key(obj):
    return json.dumps(obj)


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--result-file', type=str, default='result.jsonl')
    parser.add_argument('--task-name', type=str, default='hellaswag')
    parser.add_argument('--model-type', type=str, default='opt')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--num-fewshot', type=int, default=0)
    args = parser.parse_args()
    
    if args.model_type == 'opt':
        os.environ['MODEL_NAME'] = "facebook/opt-66b"
    elif args.model_type == 'bloom':
        os.environ['MODEL_NAME'] = "bigscience/bloom"
    elif args.model_type == 'gpt_neox':
        os.environ['MODEL_NAME'] = "EleutherAI/gpt-neox-20b"
    elif args.model_type == 'llama':
        os.environ['MODEL_NAME'] = "huggyllama/llama-7b"
    else:
        assert False

    seq = 1024
    total_batch = 1
    pe = 'fixed'

    class RealRunner:
        
        def __init__(self, args):
            
            self.results = {}
            
            with open(args.result_file, 'r') as f:
                
                for line in f:
                    if line.strip() == '':
                        continue
                    
                    item = json.loads(line)
                    
                    request = item['request']
                    result = item['result']
                    
                    self.results[json_to_key(request)] = result
            
            print(f"{len(self.results)} items in the cache")
        
        def eval(self, batch):
            
            from tasks.eval_harness import tokenizer
            
            mask_loss = []
            each_correct = []

            for i, text in enumerate(batch['text']):
                
                request = {
                        "best_of": 1, 
                        "echo": True, 
                        "logprobs": 1, 
                        "max_tokens": 0, 
                        "model": "x", 
                        "n": 1, 
                        "prompt": text, 
                        "request_type": "language-model-inference", 
                        "stop": None, 
                        "temperature": 0, 
                        "top_p": 1
                    }
                
                key = json_to_key(request)
                
                correct = True
                
                if key in self.results:
                    result = self.results[key]
                    
                    token_logprobs = result['choices'][0]['logprobs']['token_logprobs']
                    tokens = result['choices'][0]['logprobs']['tokens']
                    top_logprobs = result['choices'][0]['logprobs']['top_logprobs']
                    assert token_logprobs[0] is None
                    
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    
                    obs = batch['obs'][i]
                    target = batch['target'][i]
                    eval_mask = batch['eval_mask'][i]
                    
                    n_positive = 0
                    sum_lobprob = 0
                    if args.debug:
                        print(target)
                    for i, mask in enumerate(eval_mask):
                        try:
                            
                            if i+1 >= len(tokens):
                                break
                            
                            if mask == True:
                                if args.debug:
                                    print(tokens[i+1], next(iter(top_logprobs[i+1].keys())))
                                correct = correct and (tokens[i+1] == next(iter(top_logprobs[i+1].keys())))
                                sum_lobprob += token_logprobs[i+1]
                                n_positive += 1
                        except Exception as e:
                            raise e
                    
                    # avg_logprob = sum(token_logprobs[1:]) / (len(token_logprobs) - 1)
                    avg_logprob = sum_lobprob / n_positive
                    
                    mask_loss.append( - avg_logprob)
            
                    each_correct.append( correct )
                    
                else:
                    assert False
                

            out = {
                'mask_loss': mask_loss,
                'each_correct': each_correct,
            }
            
            
            return out

    t = RealRunner(args)

    adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")

    results = evaluator.evaluate(adaptor, tasks.get_task_dict([args.task_name
                                                               #"lambada_openai",
                                                               #"piqa",
                                                               #"hellaswag",
                                                               #"winogrande",
                                                               #"mathqa",
                                                               #"pubmedqa",
                                                               # "boolq",
                                                               # "cb",
                                                               # "copa",
                                                               # "multirc",
                                                               # "record",
                                                               # "wic",
                                                               # "wsc",
                                                               ]), False, args.num_fewshot, None)
    
    dumped = json.dumps(results, indent=2)
    print(dumped)