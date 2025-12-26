import os
import json
import argparse
import numpy as np

from utils.metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

# 对不同的数据集采用不同的评价标准
dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# 添加命令行参数
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--path', type=str, default=None)
    # 如果命令行里有--e，则e的值为True,如果没有,则为false
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--ow', action='store_true', help="Weather to overwrite existing output")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": [], '0-4k_num': 0,'4-8k_num': 0, '8k+_num': 0}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        # 算出最大得分
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        # 把对应长度的得分添加到字典里
        if length < 4000:
            scores["0-4k"].append(score)
            scores["0-4k_num"] += 1
        elif length < 8000:
            scores["4-8k"].append(score)
            scores["4-8k_num"] += 1
        else:
            scores["8k+"].append(score)
            scores["8k+_num"] += 1
    for key in ["0-4k", "4-8k", "8k+"]:
        # 计算每一个长度的平均得分，将得分转换为百分比，2为保留小数点后两位
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    # 返回平均得分
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    path = args.path
    all_files = os.listdir(path)
    out_path = f"{path}/result.json"
    if not args.ow and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
    # 遍历目录下所有数据集，计算得分
    for filename in all_files:
        
        if not filename.endswith("jsonl"):
            continue
        dataset = filename.split('.')[0]
        if not args.ow and dataset in scores:
            print(f"Dataset {dataset} already processed, skipping...")
            continue

        print(dataset)
        predictions, answers, lengths = [], [], []
        with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # predictions是列表，其中的每一项是字符串
                predictions.append(data["pred"])
                # answers是列表，其中的每一项是列表
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    
    # 把分数写入文件
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)