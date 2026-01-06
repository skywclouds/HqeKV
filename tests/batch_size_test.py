import argparse
from datasets import load_dataset
import json
import numpy as np
import os
import random
import time
import torch

from transformers import LlamaConfig, AutoTokenizer
from model.modeling_llama import LlamaForCausalLM

from flexible_quant.flexible_quantized_cache import (
    FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache)

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 对prompt进行格式化
def build_chat(tokenizer, prompt, model_name):
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        # 去掉空格和Assistant
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

# 获取模型的生成结果
def get_pred(model, tokenizer, input, model_name, batch_size):
    past_key_values = FlexibleVanillaQuantizedCache(cache_config=cache_config) if 'kvtuner' in strategy_name else None
    token_num = 32
    output = model.generate(
        **input,
        past_key_values=past_key_values,
        min_new_tokens=token_num,
        max_new_tokens=token_num,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )
    return output

def get_tokenized_input(dataset, input_length, tokenizer):
    input_text = ''
    for data in dataset:
        input_text = input_text + data['text']
        if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
            messages = [
                {"role": "user", "content": input_text}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = [text for _ in range(batch_size)]
            tokenizer.pad_token = tokenizer.eos_token
            tokenized_input = tokenizer(prompt, padding=True, truncation=False, return_tensors="pt")
        token_num = tokenized_input['input_ids'].shape[1]
        if token_num >= input_length:
            break
    return tokenized_input

def get_truncated_tokenized_input(tokenized_input, input_length, device):
    input_ids = tokenized_input['input_ids'][:, :input_length].to(device)
    attention_mask = tokenized_input['attention_mask'][:, :input_length].to(device)
    return input_ids, attention_mask

def reconstruct_tokenized_input(input_ids, attention_mask):
    tokenized_input = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        }
    return tokenized_input

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--strategies_id", type=int, default=0)
    args = parser.parse_args()
    seed_everything(42)
    device = torch.device(f'cuda:{args.gpu_id}')
    # device = torch.device('cuda:3')
    model_name_or_path = 'Llama-3.1-8B-Instruct'
    model_name =model_name_or_path.split("/")[-1]
    #                      0       1        2          3          4      5     6      7      8
    strategies =        ['hqe', 'kivi', 'kvtuner', 'zipcache', 'FP16']
    input_length_list = [4096,   32768,  36864,     61440,      65536, 81920, 131072]
    strategy_name = strategies[args.strategies_id]
    # strategy_name = strategies[0]
    batch_size = args.batch_size
    # batch_size = 50
    input_length = input_length_list[0]
    # input_length = 161
    dtype = torch.float16
    
    # 导入config和tokenizer
    if 'llama' in model_name_or_path.lower() or 'longchat' in model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if 'hqe' in strategy_name:
        # hqe 在上下文长度为 4096 时最大 batch size 为 
        config.quant_strategy = 'high_uniform_group_low_normal_group'
        # config.quant_strategy = 'uniform_group'
        config.bit_4 = 0.4159028599860191
        config.bit_2 = 0.567606203768951  
        config.bit_1 = 0.0011761525180214727
        config.bit_0 = 0.01531478372700843
        config.times_range = True
        from model.llama_hqe import LlamaForCausalLM_hqe as LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
    elif 'kivi' in strategy_name:
        from model.llama_kivi import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
    elif 'kvtuner' in strategy_name:
        cache_config = FlexibleQuantizedCacheConfig(
            nbits_key=4, nbits_value=4, asym=True, axis_key=1, axis_value=0, q_group_size=32, device=device, 
            per_layer_quant=True, per_layer_config_path='config/Llama-3.1-8B-Instruct_325.yaml')
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
    elif 'zipcache' in strategy_name:
        from model.llama_zipcache import LlamaForCausalLM
        compress_config = {}
        ## Key compress config
        compress_config["compress_mode"] = "mixed_channelwiseQ"
        compress_config["quantize_bit_important"] = 4
        compress_config["quantize_bit_unimportant"] = 2
        compress_config["k_unimportant_ratio"] = 0.4
        ## Value compress config
        compress_config["v_compress_mode"] = "channel_separate_mixed_tokenwiseQ"
        compress_config["v_quantize_bit_important"] = 4
        compress_config["v_quantize_bit_unimportant"] = 2
        compress_config["v_unimportant_ratio"] = 0.4
        compress_config["stream"] = True # streaming-gear set to true to perform better efficiency
        compress_config["streaming_gap"] = 128 # re-compress every N iteration
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            compress_config=compress_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
    else:
        # 全精度在上下文长度为 4096 时最大 batch size 为 
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
            attn_implementation="flash_attention_2",
        )

    model.eval()

    dataset = "qasper"
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    file_path = f"LongBench/{dataset}.jsonl"
    data = load_dataset('json', data_files=file_path, split='train')

    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]

    wikitext2 = load_dataset('wikitext/wikitext-103-raw-v1', split='train')
    tokenized_input = get_tokenized_input(wikitext2, input_length, tokenizer)
    input_ids, attention_mask = get_truncated_tokenized_input(tokenized_input, input_length, device)
    tokenized_input = reconstruct_tokenized_input(input_ids, attention_mask)
    
    torch.cuda.empty_cache()
    start_time = time.time()
    preds = get_pred(model, tokenizer, tokenized_input, model_name, batch_size)
    end_time = time.time()
    print(strategy_name, 'input_length:', input_length, 'batch_size:', batch_size)
    print('memory:', torch.cuda.max_memory_allocated(device) / 1024**3, 'GB')
    decoding_time = 338 * [0.0]
    for i in range(338):
        decoding_time[i] += model.model.decoding_time[i]
    prefilling_time = decoding_time[0]
    print('prefilling time:', prefilling_time)
    decoding_time = decoding_time[1:]
    decoding_time = [x for x in decoding_time if x > 0]
    if len(decoding_time) > 0:
        print('avg time per token:', sum(decoding_time)/len(decoding_time))
    