import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# 使用一个标准的 tokenizer，例如 Llama-2 或 GPT2
TOKENIZER_NAME = "gpt2" 
MAX_SEQ_LEN = 512 # 每个 chunk 的最大长度
MAX_CHUNKS = 16   # 每个问题最多保留多少个 chunk (总上下文长度 ≈ MAX_SEQ_LEN * MAX_CHUNKS)

def load_longbench_data(data_dir: str, task_names: List[str] = None):
    """加载 LongBench 数据 (JSON 格式)"""
    all_data = []
    
    # 如果没指定任务，加载目录下所有 .jsonl 或 .json 文件
    files = os.listdir(data_dir)
    for f in files:
        if not f.endswith('.jsonl') and not f.endswith('.json'):
            continue
            
        task_name = f.split('.')[0]
        if task_names and task_name not in task_names:
            continue
            
        file_path = os.path.join(data_dir, f)
        print(f"Loading {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as fin:
            # LongBench 数据通常是 JSONL 或包含 list 的 JSON
            try:
                # 尝试读取整个文件为 JSON
                content = json.load(fin)
                if isinstance(content, list):
                    for item in content:
                        item['task'] = task_name
                        all_data.append(item)
                else:
                    # 单个对象
                    content['task'] = task_name
                    all_data.append(content)
            except json.JSONDecodeError:
                # 尝试按行读取 (JSONL)
                fin.seek(0)
                for line in fin:
                    if line.strip():
                        item = json.loads(line)
                        item['task'] = task_name
                        all_data.append(item)
                        
    return all_data

def process_data(data_list, tokenizer, output_dir, split="train"):
    """
    将数据转换为 Tensor 格式并保存
    格式:
    - inputs: [N, MAX_CHUNKS, MAX_SEQ_LEN] (Context chunks)
    - query:  [N, MAX_SEQ_LEN] (Question)
    - labels: [N, MAX_SEQ_LEN] (Answer)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    inputs_list = []
    query_list = []
    labels_list = []
    meta_list = [] # 存储 task, id 等元数据

    print(f"Processing {len(data_list)} examples for {split}...")

    for idx, item in enumerate(tqdm(data_list)):
        context = item.get('context', '')
        question = item.get('input', '') # LongBench v1 use 'input' as question usually
        if not question:
            question = item.get('question', '')
            
        answer = item.get('answers', [])
        if isinstance(answer, list) and len(answer) > 0:
            answer = answer[0] # 取第一个参考答案
        elif isinstance(answer, str):
            pass
        else:
            answer = ""

        # 1. Tokenize Question (Query)
        q_tokens = tokenizer.encode(question, add_special_tokens=False, max_length=MAX_SEQ_LEN, truncation=True)
        # Pad query
        q_tensor = torch.full((MAX_SEQ_LEN,), tokenizer.eos_token_id, dtype=torch.long)
        q_tensor[:len(q_tokens)] = torch.tensor(q_tokens, dtype=torch.long)
        
        # 2. Tokenize Answer (Target)
        a_tokens = tokenizer.encode(answer, add_special_tokens=False, max_length=MAX_SEQ_LEN, truncation=True)
        # Pad answer
        a_tensor = torch.full((MAX_SEQ_LEN,), -100, dtype=torch.long) # -100 for ignore index in loss
        a_tensor[:len(a_tokens)] = torch.tensor(a_tokens, dtype=torch.long)

        # 3. Tokenize Context (Chunks)
        # 这里的策略是：把 context 切分成多个 chunks
        ctx_tokens = tokenizer.encode(context, add_special_tokens=False)
        
        chunks_tensor = torch.full((MAX_CHUNKS, MAX_SEQ_LEN), tokenizer.eos_token_id, dtype=torch.long)
        
        # 滑动窗口或直接切分
        stride = MAX_SEQ_LEN
        num_chunks = min(len(ctx_tokens) // stride + 1, MAX_CHUNKS)
        
        for i in range(num_chunks):
            start = i * stride
            end = min(start + MAX_SEQ_LEN, len(ctx_tokens))
            chunk = ctx_tokens[start:end]
            chunks_tensor[i, :len(chunk)] = torch.tensor(chunk, dtype=torch.long)
            
        inputs_list.append(chunks_tensor)
        query_list.append(q_tensor)
        labels_list.append(a_tensor)
        meta_list.append({"task": item['task'], "id": idx})

    # Stack and Save
    # 注意：Tree-TRM 原始输入是 [Batch, SeqLen]
    # 我们这里需要一种特殊的格式适配。
    # 为了复用现有 DataLoader，我们将 Chunk 视为 Puzzle Identifiers 的变体，或者直接把它们拼起来。
    # 但由于 Tree-TRM 接受单一输入，我们需要决定如何喂数据。
    
    # 策略调整：
    # Tree-TRM 的输入是 `inputs`。在数独里它是 81 个数字。
    # 在这里，我们将 `query` 作为主要的输入序列 (类似 Prompt)。
    # 将 `chunks` 作为类似于 "Puzzle Embeddings" 的外部知识库 (Memory)。
    # 所以保存为：
    # inputs -> query (Question)
    # puzzle_identifiers -> chunks (Context Memory)
    
    inputs = torch.stack(query_list) # [N, SEQ_LEN]
    puzzle_identifiers = torch.stack(inputs_list) # [N, MAX_CHUNKS, SEQ_LEN]
    labels = torch.stack(labels_list) # [N, SEQ_LEN]
    
    torch.save(inputs, os.path.join(output_dir, f"{split}_inputs.pt"))
    torch.save(puzzle_identifiers, os.path.join(output_dir, f"{split}_context.pt"))
    torch.save(labels, os.path.join(output_dir, f"{split}_labels.pt"))
    
    print(f"Saved processed data to {output_dir}")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Context shape: {puzzle_identifiers.shape}")

if __name__ == "__main__":
    # 使用 GPT2 tokenizer 作为示例 (因为不需要联网下载 huge model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    data_dir = "data/LongBench" # 原始 LongBench 数据目录
    output_dir = "data/LongBench_processed"
    
    # 1. 读取所有任务数据
    # 为了演示，我们可以只选几个代表性任务，或者全部
    # task_names = ["hotpotqa", "2wikimqa", "musique"] # Multi-hop QA 任务
    task_names = None # 全部
    
    raw_data = load_longbench_data(data_dir, task_names)
    
    # 2. 简单的 Train/Val 划分 (LongBench 只有测试集，我们划分一部分作为伪训练集)
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(raw_data)
    
    split_idx = int(len(raw_data) * 0.9)
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]
    
    # 3. 处理并保存
    process_data(train_data, tokenizer, output_dir, split="train")
    process_data(val_data, tokenizer, output_dir, split="val")
