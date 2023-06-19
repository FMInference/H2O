#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models
"""


import argparse
import logging

import numpy as np
import torch
import json
import tqdm 
import copy 

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
from utils_hh.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent, GPTNeoXAttention_Mask
from utils_hh.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
    "opt": convert_kvcache_opt_heavy_recent,
    "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_arch", type=str, default='llama')
    parser.add_argument("--model_name", type=str, default='huggyllama/llama-13b')
    parser.add_argument("--cache_dir", type=str, default='../../checkpoint/')

    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)

    parser.add_argument("--length", type=int, default=64)

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")
    set_seed(args)

    # Change to your custom prompt text
    prompt_text = 'As I walked along the winding path, the rustling of leaves filled the air, creating a soothing melody that accompanied my every step. Sunlight danced through the canopy above, casting playful shadows on the forest floor. The scent of pine mingled with the earthy aroma of damp moss, creating an intoxicating blend that awakened my senses. I paused for a moment, closing my eyes, and allowed the serenity of nature to envelop me. In this tranquil oasis, time seemed to stand still, and all worries and troubles melted away, replaced by a profound sense of peace and harmony.'
    # prompt_text = 'As I sit here on my porch, sipping my coffee and watching the world go by, I can not help but feel a sense of wonder at the sheer complexity of everything around us. From the smallest particle to the grandest galaxy, the universe is a tapestry of infinite detail and beauty. And yet, for all its complexity, there is a simplicity to it all that is truly awe-inspiring. Everything is connected, in ways that we can not even begin to fathom. Every action has a reaction, every cause has an effect. And yet, even with all the knowledge that we have amassed, there is still so much that we do not understand. There are mysteries that have eluded us for centuries, and may continue to do so for centuries to come. But that does not stop us from trying to unravel them. It does not stop us from exploring the depths of our own consciousness, or the vast expanse of the cosmos. It does not stop us from seeking answers to the biggest questions of all. Who are we? Why are we here? What is the meaning of life? These are questions that have plagued us since the dawn of time, and yet we continue to search for answers. Perhaps it is in the search itself that we find meaning. Perhaps it is in the journey, rather than the destination, that we discover the true nature of our existence. And so, as I sit here on my porch, watching the world go by, I am content to simply marvel at the beauty and complexity of it all, and to embrace the mystery that lies at the heart of our being.'

    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    config.heavy_ratio = args.heavy_ratio
    config.recent_ratio = args.recent_ratio

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=args.cache_dir)

    ######## Generate with Full Cache
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)
    model.half().eval().cuda()

    # input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(model.device)
    input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

    generate_ids = model.generate(input_ids, max_new_tokens=args.length)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("################## Generated Context with Full Cache ###################")
    print(result)


    ######### Enable HH
    checkpoint = copy.deepcopy(model.state_dict())
    model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_arch](model, config)
    model.load_state_dict(checkpoint)
    model.half().eval().cuda()

    generate_ids_hh = model.generate(input_ids, max_new_tokens=args.length)
    result_hh = tokenizer.batch_decode(generate_ids_hh, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("################## Generated Context with Heavy Hitter Oracle ###################")
    print(result_hh)


if __name__ == "__main__":
    main()