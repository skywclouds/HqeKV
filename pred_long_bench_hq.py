import argparse
from datasets import load_dataset
import json
import numpy as np
import os
import random
import torch
from accelerate.utils import tqdm
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# This is the customized building prompt for chat models
# 对prompt进行格式化
def build_chat(tokenizer, prompt, model_name):
    # For results in KIVI paper (Llama, Llama-Chat, Mistral-7B-v0.1), we do not apply any special treatment to the prompt.
    # For lmsys/longchat-7b-v1.5-32k and mistralai/Mistral-7B-Instruct-v0.2, we need to rewrite the prompt a little bit.
    # Update: we add the template for the new llama-3-instruct model
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif 'qwen' in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]   
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        # 去掉空格和Assistant
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_text_length(tokenizer, data, max_length, prompt_format: str):
    text_length = 0
    for json_obj in data:
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        prompt_length = len(tokenized_prompt)
        text_length = max(prompt_length, text_length)
    text_length = min(max_length, text_length)

    return text_length

# 获取模型的生成结果
def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, model_name):
    preds = []
    # 把输入一个一个输入进模型
    for json_obj in tqdm(data):
        # 将prompt_format中{}的内容填充成具体的内容
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if "chatglm3" in model:
        #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        # 输入超出了最大长度就分成两部分
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        # print(pred)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
        # break
    return preds


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
    parser.add_argument("--gpu_id", type=int, default=5)
    args = parser.parse_args()
    seed_everything(42)
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    
    gpu_id = args.gpu_id
    device = torch.device(f'cuda:{gpu_id}')
    # device = torch.device('cuda:2')
 
    model_name_or_path = 'Llama-3.1-8B-Instruct'
    # model_name_or_path = 'Qwen3-8B'
    e = False
    model_name =model_name_or_path.split("/")[-1]
    strategy_name = 'hqe'
    dtype = torch.float16
    
    # 导入config和tokenizer
    if 'llama' in model_name_or_path.lower() or 'longchat' in model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_name_or_path)
       
        # config._attn_implementation = "eager"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    elif 'qwen' in model_name_or_path.lower():
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        config = Qwen3Config.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    config.quant_strategy = 'high_uniform_group_low_normal_group'
    # config.quant_strategy = 'uniform_group'
    config.bit_4 = 0.25
    config.bit_2 = 0.25
    config.bit_1 = 0.25
    config.bit_0 = 0.25
    config.times_range = True
    
    if 'llama' in model_name_or_path.lower() or 'longchat' in model_name_or_path.lower():
        if 'hq' in strategy_name:
            from model.llama_hqe import LlamaForCausalLM_hqe as LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                config=config,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=device,
                attn_implementation="flash_attention_2",
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                config=config,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=device,
                attn_implementation="flash_attention_2",
            )
    elif 'qwen' in model_name_or_path.lower():
        if 'hq' in strategy_name:
            from model.qwen3_hqe import Qwen3ForCausalLM
            model = Qwen3ForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                config=config,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=device,
                attn_implementation="flash_attention_2",
            )
    else:
        raise NotImplementedError

    model.eval()
    max_length = model2maxlen[model_name]
    if e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["qasper", "narrativeqa", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    for dataset in datasets:
        if e:
            print(dataset)
            file_path = f"LongBench/{dataset}_e.jsonl"
            data = load_dataset('json', data_files=file_path, split='train')
            if not os.path.exists(f"pred_e/{model_name}_{max_length}_{strategy_name}"):
                os.makedirs(f"pred_e/{model_name}_{max_length}_{strategy_name}")
            out_path = f"pred_e/{model_name}_{max_length}_{strategy_name}/{dataset}.jsonl"
        else:
            print(dataset)
            file_path = f"LongBench/{dataset}.jsonl"
            data = load_dataset('json', data_files=file_path, split='train')
            if not os.path.exists(f"pred/{model_name}_{max_length}_{strategy_name}"):
                os.makedirs(f"pred/{model_name}_{max_length}_{strategy_name}")
            out_path = f"pred/{model_name}_{max_length}_{strategy_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
            
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
        # print(preds)
        # break
        
