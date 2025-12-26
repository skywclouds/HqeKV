import argparse
from datasets import load_dataset
from model.llama_ratio_search import LlamaForCausalLM_hqe
from model.qwen3_ratio_search import Qwen3ForCausalLM
import optuna
import os
import torch
from transformers import AutoTokenizer, LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def func(x_3, x_4):
    bit_16 = bit_8 = 0
    bit_4 = (avg_bit - 2 + x_3 + 2*x_4) / 2
    bit_2 = (4 - avg_bit - 3*x_3 - 4*x_4) / 2
    bit_1 = x_3
    bit_0 = x_4
    if bit_4 < 0 or bit_2 < 0:
        loss = float('inf')
    else:
        compress_ratio = torch.tensor([bit_16, bit_8, bit_4, bit_2, bit_1, bit_0]) 
        with torch.no_grad():
            output = model(**tokenized_input, compress_ratio=compress_ratio)
            quant_output = output.logits.squeeze(0)
            loss = model.loss_function(logits=quant_output, labels=full_output, vocab_size=model.config.vocab_size)
            loss = float(loss) + + 2*x_3 + 4*x_4
    return loss


def objective(trial):
    x_3 = trial.suggest_float('x_3', 0.0, 1.0)
    x_4 = trial.suggest_float('x_4', 0.0, 1.0)
    return func(x_3, x_4)        # 损失越小越好

def get_tokenized_input(dataset, input_length, tokenizer):
    input_text = ''
    for data in dataset:
        input_text = input_text + data['text']
        if 'qwen' in model_name.lower():
            messages = [
                {"role": "user", "content": input_text}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            tokenized_input = tokenizer(text, truncation=False, return_tensors="pt")
        else:
            tokenized_input = tokenizer(input_text, truncation=False, return_tensors="pt")
        token_num = tokenized_input['input_ids'].shape[1]
        if token_num >= input_length:
            break
    return tokenized_input

def get_truncated_tokenized_input(tokenized_input, input_length, device):
    input_ids = tokenized_input['input_ids'][:, :input_length].to(device)
    attention_mask = tokenized_input['attention_mask'][:, :input_length].to(device)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels

def reconstruct_tokenized_input(input_ids, attention_mask):
    tokenized_input = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        }
    return tokenized_input

def get_full_output(model, tokenized_input):
    compress_ratio = torch.tensor([1, 0, 0, 0, 0, 0])
    with torch.no_grad():
        output = model(**tokenized_input, compress_ratio=compress_ratio)
    # full_output = output.logits[0, -1, :].to(torch.float32)
    full_output = torch.argmax(output.logits, dim=-1)
    full_output.squeeze_(0)
    return full_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--avg_bit", type=float, default=2.0)
    parser.add_argument("--gpu_id", type=int, default=6)
    args = parser.parse_args()

    wikitext2 = load_dataset('wikitext/wikitext-103-raw-v1', split='train')

    model_name = 'llama'
    # model_name = 'qwen3'

    if 'llama' in model_name.lower():
        model_name_or_path = 'Llama-3.1-8B-Instruct'
        config = LlamaConfig.from_pretrained(model_name_or_path)
    elif 'qwen' in model_name.lower():
        model_name_or_path = 'Qwen3-8B'
        config = Qwen3Config.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    gpu_id = args.gpu_id
    device = torch.device(f'cuda:{gpu_id}')
    dtype = torch.float16
    input_length = 16 * 1024

    tokenized_input = get_tokenized_input(wikitext2, input_length, tokenizer)
    input_ids, attention_mask, labels = get_truncated_tokenized_input(tokenized_input, input_length, device)
    tokenized_input = reconstruct_tokenized_input(input_ids, attention_mask)

    model = None
    if 'llama' in model_name.lower():
        model = LlamaForCausalLM_hqe.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
    elif 'qwen' in model_name.lower():
        model = Qwen3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
    avg_bit = args.avg_bit
    full_output =  get_full_output(model, tokenized_input)
    

    # 3. 创建并运行研究
    study = optuna.create_study(direction='minimize', load_if_exists=True)   # minimize 表示损失最小化
    study.optimize(objective, n_trials=200)

    # 4. 查看结果
    x_3 = study.best_params['x_3']
    x_4 = study.best_params['x_4']
    bit_4 = (avg_bit - 2 + x_3 + 2*x_4) / 2
    bit_2 = (4 - avg_bit - 3*x_3 - 4*x_4) / 2
    bit_1 = x_3
    bit_0 = x_4

    print('最优参数:', bit_4, bit_2, bit_1, bit_0)
    print('最小损失:', study.best_value)
