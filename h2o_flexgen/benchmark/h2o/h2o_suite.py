import argparse
from dataclasses import dataclass

from flexgen.utils import run_cmd


@dataclass
class Case:
    command: str
    name: str = ""
    use_page_maga: bool = False


suite_flexgen = [
    # FlexGen
    # opt-6.7b
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 2 --prompt-len 512 --gen-len 32 --cut-gen-len 8"),
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 1 --prompt-len 512 --gen-len 512 --cut-gen-len 128"),
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 1 --prompt-len 512 --gen-len 1024 --cut-gen-len 128"),

    # opt-30b
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 20 80 0 100 0 100 --gpu-batch-size 40 --num-gpu-batches 2 --cpu --debug fewer_batch --gen-len 512"),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 1 --cpu --debug fewer_batch --gen-len 1024"),
]


suite_h2o_20 = [
    # H2O
    # opt-6.7b
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 4 --prompt-len 512 --gen-len 32 --cut-gen-len 8 --hh-ratio 0.1 --hh-all"),
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 4 --prompt-len 512 --gen-len 512 --cut-gen-len 128 --hh-ratio 0.2 --hh-all"),
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 4 --prompt-len 512 --gen-len 1024 --cut-gen-len 128 --hh-ratio 0.3 --hh-all"),

    # opt-30b
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 52 --num-gpu-batches 14 --cpu --prompt-len 512 --gen-len 32 --debug fewer_batch --hh-ratio 0.1 --hh-all"),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 52 --num-gpu-batches 8 --cpu --prompt-len 512 --gen-len 512 --debug fewer_batch --hh-ratio 0.2 --hh-all"),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 44 --num-gpu-batches 6 --cpu --prompt-len 512 --gen-len 1024 --debug fewer_batch --hh-ratio 0.3 --hh-all"),

    # H2O weights compress
    # opt-6.7b
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 70 --prompt-len 512 --gen-len 32 --hh-ratio 0.1 --hh-all --compress-weight"),
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 52 --prompt-len 512 --gen-len 512 --cut-gen-len 128 --hh-ratio 0.2 --hh-all --compress-weight"),
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 44 --prompt-len 512 --gen-len 1024 --cut-gen-len 128 --hh-ratio 0.3 --hh-all --compress-weight"),
]

suite_h2o_20_a100_80 = [
    # FlexGen
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 1 --prompt-len 7000 --gen-len 1024 --hh-long-seq"),
    Case("--model facebook/opt-13b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 4 --prompt-len 5000 --gen-len 5000 --hh-long-seq"),
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 24 --prompt-len 2048 --gen-len 2048 --hh-long-seq"),

    # H2O
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 1 --prompt-len 7000 --gen-len 1024 --hh-long-seq --hh-ratio 0.11 --hh-all"),
    Case("--model facebook/opt-13b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 4 --prompt-len 5000 --gen-len 5000 --hh-long-seq --hh-ratio 0.2 --hh-all"),
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 24 --prompt-len 2048 --gen-len 2048 --hh-long-seq --hh-ratio 0.2 --hh-all"),

    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --overlap False --gpu-batch-size 64 --prompt-len 2048 --gen-len 2048 --hh-long-seq --hh-ratio 0.2 --hh-all"),
]

suite_test = [
]

suites = {
    "flexgen": suite_flexgen,
    "h2o_20": suite_h2o_20,
    "h2o_20_a100_80": suite_h2o_20_a100_80,
    "test": suite_test,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=str, nargs="+")
    parser.add_argument("--log-file", type=str)
    args = parser.parse_args()

    log_file = args.log_file

    for suite in args.suite:
        cases = suites[suite]
        for case in cases:
            config, name, use_page_maga = case.command, case.name, case.use_page_maga
            cmd = f"python -m flexgen.flex_opt {config}"
            if log_file:
                cmd += f" --log-file {args.log_file}"
            if use_page_maga:
                cmd = "bash /usr/local/bin/pagecache-management.sh " + cmd

            if log_file:
                with open(log_file, "a") as f: f.write(f"#### {name}\n```\n{cmd}\n")
            run_cmd(cmd)
            if log_file:
                with open(log_file, "a") as f: f.write(f"```\n")
