import argparse
from dataclasses import dataclass
import time

from flexgen.utils import run_cmd


def run_huggingface(model, prompt_len, gen_len, cut_gen_len, batch_size,
                    num_nodes, num_gpus_per_node,
                    use_ds, cpu, disk, dummy, log_file=None, pkl_file=None):
    assert num_nodes == 1
    if use_ds:
        cmd = f"deepspeed --num_gpus {num_gpus_per_node} hf_opt.py "
    else:
        cmd = f"python hf_opt.py --num-gpus {num_gpus_per_node} "

    cmd += (f"--model {model} "
            f"--prompt-len {prompt_len} --gen-len {gen_len} "
            f"--batch-size {batch_size} ")

    if cut_gen_len:
        cmd += f"--cut-gen-len {cut_gen_len} " 
    if cpu:
        cmd += "--cpu "
    if disk:
        cmd += "--disk "
    if dummy:
        cmd += "--dummy "

    if log_file is not None:
        cmd += f"--log-file {log_file} "
    if pkl_file is not None:
        cmd += f"--pkl-file {pkl_file} "

    run_cmd(cmd)


def bench_one_case(case):
    if case.model == "facebook/opt-6.7b" and case.device == "gpu":
        cut_gen_len = None
    else:
        cut_gen_len = 16
    dummy = True

    if case.device == "gpu":
        cpu = disk = False
    elif case.device == "cpu":
        cpu, disk = True, False
    elif case.device == "disk":
        cpu, disk = False, True

    use_deepspeed = case.library == "ds"

    run_huggingface(case.model, case.prompt_len, case.gen_len, cut_gen_len,
                    case.batch_size, case.num_nodes, case.num_gpus_per_node,
                    use_ds=use_deepspeed,
                    cpu=cpu, disk=disk, dummy=dummy)


@dataclass
class Case:
    model: str
    library: str
    prompt_len: int
    gen_len: int
    batch_size: int
    device: str
    num_nodes: int = 1
    num_gpus_per_node: int = 1


# h2o suite
suite_hf_h2o = [
    Case("facebook/opt-6.7b", "hf", 512, 32, 2, "gpu"),
    Case("facebook/opt-6.7b", "hf", 512, 512, 1, "gpu"),
    Case("facebook/opt-6.7b", "hf", 512, 1024, 16, "cpu"),

    Case("facebook/opt-30b", "hf", 512, 32, 8, "cpu"),
    Case("facebook/opt-30b", "hf", 512, 512, 8, "cpu"),
    Case("facebook/opt-30b", "hf", 512, 1024, 8, "cpu"),
]

suite_ds_h2o = [
    Case("facebook/opt-6.7b", "ds", 512, 32, 16, "cpu"),
    Case("facebook/opt-6.7b", "ds", 512, 512, 16, "cpu"),
    Case("facebook/opt-6.7b", "ds", 512, 1024, 16, "cpu"),

    Case("facebook/opt-30b", "ds", 512, 32, 4, "cpu"),
    Case("facebook/opt-30b", "ds", 512, 512, 4, "cpu"),
    Case("facebook/opt-30b", "ds", 512, 1024, 4, "cpu"),
]

suites = {
    "hf_h2o": suite_hf_h2o,
    "ds_h2o": suite_ds_h2o,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=str, nargs="+")
    args = parser.parse_args()

    cases = []
    for suite in args.suite:
        cases += suites[suite]

    for case in cases:
        tic = time.time()
        bench_one_case(case)
        print(f"elapsed: {time.time() - tic:.2f} s")
        time.sleep(2)
