import argparse

import numpy as np
import torch
import json
import tqdm 
import copy 

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def get_policy(model_name):
    if model_name == "opt-6.7b":
        gpu_batch_size = 1
        num_gpu_batches = 1
        percent = (100, 0, 100, 0, 100, 0)
        cpu_cache_compute = False
    elif model_name == "opt-30b":
        gpu_batch_size = 1
        num_gpu_batches = 1
        percent = (0, 100, 0, 100, 0, 100)
        cpu_cache_compute = True
    else:
        raise Exception("unsupported model")

    policy = Policy(gpu_batch_size, num_gpu_batches,
                    percent[0], percent[1],
                    percent[2], percent[3],
                    percent[4], percent[5],
                    overlap=True, sep_layer=True, pin_weight=True,
                    cpu_cache_compute=cpu_cache_compute, attn_sparsity=1.0,
                    compress_weight=False,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=False,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))
    return policy


def get_batches(requests, policy):
    batches = []
    cpu_batch_size = policy.gpu_batch_size * policy.num_gpu_batches
    num_batch = ((len(requests) - 1) // cpu_batch_size) + 1

    for i in range(num_batch):
        batch = {"max_new_tokens": 0, "input_ids": [], "eos_token_id": None}

        start_idx = i * cpu_batch_size
        if i == num_batch - 1:
            end_idx = len(requests)
            last_policy = policy.deepcopy()
            last_policy.num_gpu_batches = (end_idx - start_idx - 1) // last_policy.gpu_batch_size + 1
            new_batch_size = last_policy.gpu_batch_size * last_policy.num_gpu_batches
            for j in range(cpu_batch_size):
                requests.append(requests[end_idx - 1])
            end_idx = start_idx + new_batch_size
        else:
            end_idx = (i + 1) * cpu_batch_size
        assert (end_idx - start_idx) % policy.gpu_batch_size == 0

        prompt_len = 0
        for req in requests[start_idx : end_idx]:
            max_tokens = req["request"]["max_tokens"]
            prompt = req["request"]["prompt"]
            stop = req["request"]["stop"]
            input_ids = tokenizer([prompt], add_special_tokens=False).input_ids[0]
            eos_token_id = tokenizer(stop).input_ids.to(model.device)

            prompt_len = max(prompt_len, len(input_ids))
            batch["max_new_tokens"] = max(batch["max_new_tokens"], max_tokens)
            batch["input_ids"].append(input_ids)
            # assert batch["eos_token_id"] is None or batch["eos_token_id"] == eos_token_id
            # batch["eos_token_id"] = eos_token_id

        # manual padding
        for i in len(batch["input_ids"]):
            cnt = prompt_len - len(batch["input_ids"][i])
            batch["input_ids"][i] = np.pad(batch["input_ids"][i], (cnt, 0), "constant", constant_values=(0))

        batches.append(batch)

    return batches, last_policy


def run_inference(model_name, requests):
    assert model_name == "opt-6.7b" or model_name == "opt-30b"

    env = ExecutionEnv.create("~/flexgen_offload_dir")
    policy = get_policy(model_name)

    batches, last_policy = get_batches(requests, policy)

    print(f"Init weights begin.")
    tic = time.time()
    model = OptLM(model_name, env, "~/opt_weights", policy)
    print(f"Init weights end. Elapsed: {time.time() - tic:.2f} s", flush=True)

    # Generate
    print(f"Generate begin. #sequences: {len(batches) * effective_bs}")
    tic = time.time()
    input_ids_batches = []
    output_ids_batches = []
    gen_tokens = 0
    for i, batch in tqdm(enumerate(batches)):
        input_ids = batch["input_ids"]
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            temperature=1e-7,
            max_new_tokens=batch["max_new_tokens"],
            stop=batch["eos_token_id"])
        input_ids_batches.append(input_ids)
        output_ids_batches.append(output_ids)
        gen_tokens += len(input_ids) * batch["max_new_tokens"]
    total_time = time.time() - tic

    # last batch
    model = OptLM(model_name, env, "~/opt_weights", last_policy)
    tic = time.time()
    batch = batches[-1]
    input_ids = batch["input_ids"]
    output_ids = model.generate(
        input_ids,
        do_sample=True,
        temperature=1e-7,
        max_new_tokens=batch["max_new_tokens"],
        stop=batch["eos_token_id"])
    gen_tokens += len(input_ids) * batch["max_new_tokens"]
    total_time += time.time() - tic

    print(f"Generate end. Elapsed: {total_time:.2f} s", flush=True)
    print(f"Generation throughput: {gen_tokens / total_time:.2f} token/s", flush=True)

    input_ids = np.concatenate(input_ids_batches)
    output_ids = np.concatenate(output_ids_batches)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #print("Outputs:\n" + 70 * '-')
    ##for i in range(len(outputs)):
    #for i in [0, len(outputs) - 1]:
    #    print(f"{i}:\n{outputs[i]}")
    #    print("-" * 70)

    return outputs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="../../../h2o_hf/data/xsum_opt.jsonl")
    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint/")

    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)

    parser.add_argument("--sample_num", type=int, default=1000)

    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    args.device = torch.device("cuda")
    args.n_gpu = 1

    set_seed(args)

    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path 

    # load requests
    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))
    print(len(requests))
    if args.sample_num < len(requests):
        print('Sample {} Examples'.format(args.sample_num))
    requests = requests[:args.sample_num]
    requests = sorted(requests, key=lambda x: len(x["request"]["prompt"]))
    # print([len(req["request"]["prompt"]) for req in requests])

    # run inference
    results = run_inference(model_name, requests)
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    main()
