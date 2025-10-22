import sys
import os
import argparse
from tqdm import tqdm
import logging
import os
import time
import torch
import gc
import json
import copy
import pandas as pd

from collections import defaultdict

import numpy as np
import random
from utils import cot_prompt, kl_scoring, simi_scoring, get_intervation_result, emb_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from interveners import wrapper, Collector, ITI_Intervener, head_Intervener
import pyvene as pv
from sentence_transformers import SentenceTransformer
import csv
from datasets import load_dataset
import re
from llm import efficient_openai_text_api
from ablation_task_recorrect import scoring
import evaluate

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PL_NO_SLURM"] = "1"
os.environ["SLURM_JOB_ID"] = ""
os.environ["SLURM_ARRAY_TASK_ID"] = ""

torch.set_float32_matmul_precision('high')

HF_NAMES = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama3.1_8B_instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama3.2_3B_instruct': 'meta-llama/Llama-3.2-3B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'qwen3-8B': 'Qwen/Qwen3-8B',
    'qwen3-1.7B': 'Qwen/Qwen3-1.7B',
    'qwen3-4B': 'Qwen/Qwen3-4B',
    'gemma3-12B': 'google/gemma-3-12b-it',
    'gemma3-4B': 'google/gemma-3-4b-it',
    'phi4-3.8B': "microsoft/Phi-4-mini-instruct",
    'internlm2-1.8B': 'internlm/internlm2-1_8b',
    'yi-1.5-6B': '01-ai/Yi-1.5-6B-Chat',
    'yi-1.5-9B': '01-ai/Yi-1.5-9B-Chat'
}

# # define the intervention function
def run_intervention(inter_heads_idx):
    pv_config = []
    for layer_idx, heads in inter_heads_idx.items():
        # setting head_mask
        head_mask = torch.ones(head_dim * num_heads, dtype=torch.float32).to(model.device)
        for head in heads:
            head_mask[head * head_dim:(head + 1) * head_dim] = 0.0001
        intervener = head_Intervener(head_mask=head_mask)   
        intervener.layer_idx = layer_idx  # note: to let __call__ access current layer index
        pv_config.append({
            "component": f"model.layers[{layer_idx}].self_attn.o_proj.input",
            "intervention": wrapper(intervener),
        })

    intervened_model = pv.IntervenableModel(pv_config, model)
    
    return intervened_model


def get_intervention_heads(layer_idx, head_idx, num_layers, num_heads):
    """
    getting the intervention heads index
    """
    return {layer_idx: [head_idx]}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass

def get_topk_intervention_heads(head_features):
    """
    获取指定层和头的干预索引
    :param head_features: List[int], 每个头的特征索引
    :param num_heads: int, 每层的头数
    :return: dict, 按层索引分组的头索引
    """
    inter_heads_idx = {}
    for feature in head_features:
        if type(feature) != int:
            feature = feature[0]
        layer_idx = feature // num_heads
        head_idx = feature % num_heads
        if layer_idx not in inter_heads_idx:
            inter_heads_idx[layer_idx] = [head_idx]
        else:
            inter_heads_idx[layer_idx].append(head_idx)
    
    return inter_heads_idx

def get_last_intervention_heads(head_features):
    save_heads_idx = {}
    inter_heads_idx = {}
    for feature in head_features:
        if type(feature) != int:
            feature = feature[0]
        layer_idx = feature // num_heads
        head_idx = feature % num_heads
        if layer_idx not in save_heads_idx:
            save_heads_idx[layer_idx] = [head_idx]
        else:
            save_heads_idx[layer_idx].append(head_idx)

    for i in range(num_layers):
        inter_heads_idx[i] = list(range(num_heads))
        if i in save_heads_idx:
            for head in save_heads_idx[i]:
                inter_heads_idx[i].remove(head)
    return inter_heads_idx

def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj
    
system_role = """
You are an expert in analytical and logical reasoning. Your task is to answer the question.
"""

llama_prompt = """
{system_role}
Here is the question:
<question>
{question}
</question>

Instructions:
- Think through the problem carefully and logically.
- Provide a step-by-step explanation leading to the final answer.
- Be concise and avoid unnecessary information.    

Output format:
[
  {{
    "explanation": "<Your explanation here>",
    "final answer": "<Your final answer here>"
  }}
]

Your answer:
"""    

other_prompt = """
{system_role}
Here is the question:
<question>
{question}
</question>

Instructions:
- Think through the problem step by step to provide the explanation and the final answer.
- Be concise and avoid unnecessary information.   
- Your output must strictly follow the format shown below. 

Output format:
[
  {{
    "explanation": "<Your step-by-step thinking process here>",
    "final answer": "<Your final answer here>"
  }}
]

Your answer:
"""

extraction_prompt = """

You are a document information assistant. Please answer the user's question using only exact sentences copied from the document content. Do not paraphrase or infer beyond what is explicitly stated.

<Document Content>
{document_text}
</Document Content>

<Question>
{question}
</Question>

Extract the most relevant sentence(s) from the document that directly answer the question.

Output format:
[
  {{
    "answer": "<answer here>"
  }}
]
"""


gpt_prompt_onlyanswer = """
You are given a reference answer and a predicted answer to a question.

This is the reference answer:
<reference_answer>
{reference_answer}
</reference_answer>

This is the predicted answer:
<prediction_answer>
{prediction_answer}
</prediction_answer>

Instructions:
1. Return `"True"` if the prediction answer is consistent to the reference answer. Otherwise, return `"False"`.
2. Provide a **confidence score** between `0` and `1` based on how certain you are.

Output format:
[
{{
  "correct": <True|False>,
  "confidence": <float between 0 and 1>
}}
]
Your answer:
"""

def get_output(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_only_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()
    output_answer = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
    answer = ''
    explanation = ''
    match = re.search(r'"answer"\s*:\s*["\']?(.*?)["\']?,', output_answer)
    if match:
        answer = match.group(1)
        print("Answer:", answer)
    match = re.search(r'"explanation"\s*:\s*"(.+?)"\s*}', output_answer, re.DOTALL)
    if match:
        explanation = match.group(1).strip()
        print("Explanation:", explanation)
    else:
        print("No explanation found.")
    return answer, explanation

def input_prompt(dataset):
    all_prompts = []
    
    k = 0
    for i in range(len(dataset)):
        ref_answer = dataset[i]['answer']
        pre_answer = dataset[i]['response']["answer"]

        # format prompt
        prompt_text = gpt_prompt_onlyanswer.format(
            reference_answer=ref_answer,
            prediction_answer=pre_answer
        )
        k += 1

        all_prompts.append(prompt_text)

    return all_prompts

def get_result_from_output(outputs):
    answer_list = []
    confidence_list = []
    
    for i, output in enumerate(outputs):
        output = output[0][0].replace("True", "true").replace("False", "false")
        matches = re.findall(r'\[\s*{[^}]+}\s*\]', output)
        if matches:
            output_json_block = matches[-1].strip()
            print("Extracted JSON answer block:", output_json_block)

            try:
                # parsed = safe_json_parse(output_json_block)
                parsed = json.loads(output_json_block)
                answer, confidence = parsed[0]["correct"], parsed[0]["confidence"]  # This will return '"$80"'
                print("Final extracted answer (with quotes):", answer)
            except:
                answer_match = re.search(r'"correct":\s*"([^"]+)"', output)
                confidence_match = re.search(r'"confidence":\s*([\d.]+)', output)
                
                # 如果匹配到值则返回
                if answer_match and confidence_match:
                    answer = answer_match.group(1)
                    confidence = float(confidence_match.group(1))

                else:
                    answer = ""
                    confidence = 0
        
        answer_list.append(answer)
        confidence_list.append(confidence)
        
        # answer_list2.append(answer2)
        # confidence_list2.append(confidence2)
        
        print(f"Answer: {answer}, Confidence: {confidence}")
    
    return answer_list, confidence_list

def extract_answer(output_text):
    # 提取最后的 "#### 数字" 作为模型的答案
    match = re.search(r"####\s*(-?[0-9,\.]+)", output_text)
    if match:
        return match.group(1).replace(",", "")
    return None

def extract_number(text):
    # 提取第一个整数或小数（支持负号）
    match = re.search(r'-?\d+(\.\d+)?', text)
    return float(match.group()) if match else None

def parse_json_with_dollar(json_str):
    # using regex to replace dollar amounts with numbers
    cleaned_str = re.sub(
        r'"final answer"\s*:\s*"\$?(\d+(?:\.\d+)?)"|'
        r'"final answer"\s*:\s*\$?(\d+(?:\.\d+)?)',
        lambda m: f'"final answer": {m.group(1) or m.group(2)}',
        json_str
    )
    
    return cleaned_str

def extract_explanations_and_answers(text):
    """
    extract explanations and final answers from model output text
    """
    json_blocks = re.findall(r'\[\s*\{.*?\}\s*\]', text, flags=re.DOTALL)
    
    results = []
    for block in json_blocks:
        try:
            block = parse_json_with_dollar(block)
            data = json.loads(block)
            if isinstance(data, list) and isinstance(data[0], dict):
                explanation = str(data[0].get("explanation", "")).strip()
                final_answer = str(data[0].get("final answer", "")).strip()
                results.append({
                    "explanation": explanation,
                    "final_answer": final_answer
                })
        except Exception as e:
            print(f"error {e}")
            continue

    return results

def scoring(base_prediction, reference_answer):
    predictions = [str(base_prediction)]
    references = [str(reference_answer)]
    results = {}

    # BLEU

    bleu_score = bleu.compute(predictions=predictions, references=references)
    results["bleu"] = bleu_score["bleu"]

    # ROUGE
    rouge_score = rouge.compute(predictions=predictions, references=references)
    results.update(rouge_score)  # rouge1, rouge2, rougeL, rougeLsum
    
    cosine_score = emb_similarity([predictions, references])
    results["cosine"] = cosine_score

    # COMET
    if args.use_comet:
        comet_score = comet.compute(
            predictions=predictions,
            references=references,
            sources=references
        )
        results["comet"] = comet_score["mean_score"]
    else:
        results["comet"] = "Skipped (no sources provided)"
    return results

functions_names = ["Retrieval", "Knowledge Recall", "Semantic Understanding", "Syntactic Understanding", "Math Calculation", "Inference", "Logical Reasoning", "Decision-making"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1_8B_instruct')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='extraction_qa')
    parser.add_argument('--token_use', type=str, default="topk")
    parser.add_argument('--mask_ratio', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--llm_model_method', default = 'o4-mini', type = str)
    parser.add_argument('--key', default = 'your key', type = str)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--function_name', type=str, default='Retrieval')
    parser.add_argument('--use_bleu', default=True, action='store_true')
    parser.add_argument('--use_rouge', default=True, action='store_true')
    parser.add_argument('--use_cosine', default=True, action='store_true')
    parser.add_argument('--use_comet', default=True, action='store_true')
    parser.add_argument('--use_kl', default=False, action='store_true')
    parser.add_argument('--use_head', type=str, default="topk")
    parser.add_argument('--use_layer_bias', default=True, action="store_true")
    parser.add_argument('--output_dir', type=str, default="mask_head_acc.csv")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    with open(f"main_results/{args.model_name}/layer_True_position_topk_elbow.json", "r") as f:
        elbow_index = json.load(f)
    args.mask_ratio = elbow_index[args.function_name]

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
   
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token='your token', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, token='your token', torch_dtype=torch.float16, device_map="cuda:0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    device = "cuda"
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    if args.model_name == "qwen3-4B":
        head_dim = model.config.head_dim
    else:
        head_dim = hidden_size // num_heads    
        
    mask_config = {
        "scale_factor": 0.0001,
        "mask_type": "scale_mask",
    }
    
    
    if args.dataset_name == "cot_qa": 
        path = 'dataset/balanced_cot_test_data.json'
        dataset = json.load(open(path, "r")) 
    elif args.dataset_name == "GSM8K":
        dataset = load_dataset("gsm8k", "main") 
        dataset = dataset['test'][:100]
    elif args.dataset_name == "extraction_qa":
        path = 'dataset/extraction_qa.json'
        dataset = json.load(open(path, "r"))
            
    if os.path.exists(f"inter_results/{args.model_name}/{args.dataset_name}.json"):
        with open(f"inter_results/{args.model_name}/extra_new.json", "r") as f:
            dataset = json.load(f)
    else:
        prompts = []
        if args.dataset_name == "GSM8K":
            for i in range(len(dataset["question"])):
                question = dataset["question"][i]
                if "llama" in args.model_name:
                    prompt = llama_prompt
                else:
                    prompt = other_prompt
                text = prompt.format(
                        system_role=system_role,
                        question=question
                    )
                prompts.append(text)
        elif args.dataset_name == "extraction_qa":
            for i in range(len(dataset)):
                question = dataset[i]["question"]
                text = extraction_prompt.format(
                        document_text=dataset[i]["paragraph"],
                        question=question
                    )
                prompts.append(text)
        elif args.dataset_name == "Idavidrein/gpqa":
            for i in range(len(dataset["questions"])):
                
                question = dataset["questions"][i]["question"]
                choices = dataset["questions"][i]["choices"]
                choices_str = " ".join([f"{key}. {value}" for key, value in choices.items()])
                question = question +" "+ choices_str
                text = prompt.format(
                        system_role=system_role,
                        question=question
                    )
                prompts.append(text)
        
    print("Start Intervation")
    
    with open(f"main_results/{args.model_name}/layer_{args.use_layer_bias}_position_{args.token_use}.json", "r") as f:
        importances = json.load(f)
    if args.use_head == "randomk":
        head_features = importances[args.function_name][:args.mask_ratio]
        all_heads = list(range(num_layers * num_heads))
        all_heads = [h for h in all_heads if h not in head_features]
        head_features = random.sample(all_heads, args.mask_ratio)
    elif args.use_head == "lowk":
        head_features = importances[args.function_name][-args.mask_ratio:]
    elif args.use_head == "topk":
        head_features = importances[args.function_name][:args.mask_ratio]
    
    if args.use_head in ["topk", "randomk", "lowk"]:
        inter_heads_idx = get_topk_intervention_heads(head_features) 
    elif args.use_head == "last":
        inter_heads_idx = get_last_intervention_heads(head_features)
    
    intervened_model = run_intervention(inter_heads_idx)

    scores_list = []
    generated_answers = []
    acc_head = {}
    base_acc_num = 0
    adv_acc_num = 0
    base_results = []
    adv_results = [] 
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    comet = evaluate.load("comet", experiment_id=f"comet_{os.getpid()}_{int(time.time()*1000)}")      
    if not os.path.exists(os.path.join(f"rebuttal_results/{args.model_name}", args.function_name.replace(" ", "_"))):
        os.makedirs(os.path.join(f"rebuttal_results/{args.model_name}", args.function_name.replace(" ", "_")))
    
    for k in range(len(prompts)):
        prompt = prompts[k]
        label = args.function_name
        
        if args.dataset_name == "GSM8K":
            answer = extract_answer(dataset["answer"][k])
            explanation = dataset["answer"][k]
        elif args.dataset_name == "extraction_qa":
            answer = dataset[k]["answer"]
            explanation = dataset[k]["answer"]
        elif args.dataset_name == "Idavidrein/gpqa":
            answer = dataset["questions"][k]["correct_answer"]
            explanation = dataset["questions"][k]["explanation"]

        base_answer, adv_answer = get_intervation_result(prompt, tokenizer, intervened_model, device, args)

        if args.dataset_name == "GSM8K":
            try:
                base_answer_all = extract_explanations_and_answers(base_answer)[-1]
                adv_answer_all = extract_explanations_and_answers(adv_answer)[-1]
                base_result = scoring(base_answer_all["explanation"], explanation)
                adv_result = scoring(adv_answer_all["explanation"], explanation)
                num_base = extract_number(base_answer_all["final_answer"])
                num_adv = extract_number(adv_answer_all["final_answer"])
                num_answer = extract_number(answer)
                if num_base is not None and num_answer is not None and num_base == num_answer:
                    base_acc_num += 1
                if num_adv is not None and num_answer is not None and num_adv == num_answer:
                    adv_acc_num += 1
            except Exception as e:
                base_result = scoring(base_answer, answer)
                adv_result = scoring(adv_answer, answer)
                print(f"Error for data {k}")
        elif args.dataset_name == "extraction_qa":
            base_answer = re.findall(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', base_answer, flags=re.DOTALL)
            adv_answer = re.findall(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', adv_answer, flags=re.DOTALL)
            base_result = scoring(base_answer, answer)
            adv_result = scoring(adv_answer, answer)
        
        data_save = {"question": prompt, "correct_answer":answer, "explanation":explanation, "base_answer": base_answer, "adv_answer": adv_answer}
        with open(os.path.join(f"rebuttal_results/{args.model_name}", args.function_name.replace(" ", "_"), "mask_"+args.dataset_name+ args.use_head + "_head_answer.json"), "a") as f:
            # json.dump(data_save, f, indent=4)
            f.write(json.dumps(data_save, indent=4) + ",\n")
        base_results.append(base_result)
        adv_results.append(adv_result)
    if args.use_bleu:
        scores_bleu = [score["bleu"] for score in base_results]
        base_bleu = np.mean(scores_bleu)
        print(f"BLEU: {base_bleu:.6f}")
        scores_bleu = [score["bleu"] for score in adv_results]
        adv_bleu = np.mean(scores_bleu)
        print(f"BLEU: {adv_bleu:.6f}")
    if args.use_rouge:
        scores_rouge1 = [score["rouge1"] for score in base_results]
        base_rouge1 = np.mean(scores_rouge1)
        print(f"ROUGE: {base_rouge1:.6f}")
        scores_rouge1 = [score["rouge1"] for score in adv_results]
        adv_rouge1 = np.mean(scores_rouge1)
        print(f"ROUGE: {adv_rouge1:.6f}")
    if args.use_cosine:
        scores_cosine = [score["cosine"] for score in base_results]
        base_cosine = np.mean(scores_cosine)
        print(f"Cosine: {base_cosine:.6f}")
        scores_cosine = [score["cosine"] for score in adv_results]
        adv_cosine = np.mean(scores_cosine)
        print(f"Cosine: {adv_cosine:.6f}")
    if args.use_comet:
        scores_comet = [score["comet"] for score in base_results]
        base_comet = np.mean(scores_comet)
        print(f"COMET: {base_comet:.6f}")
        scores_comet = [score["comet"] for score in adv_results]
        adv_comet = np.mean(scores_comet)
        print(f"COMET: {adv_comet:.6f}")
    
    base_score = [{"bleu": base_bleu, "rouge": base_rouge1, "cosine": base_cosine, "comet": base_comet}]
    adv_score = [{"bleu": adv_bleu, "rouge": adv_rouge1, "cosine": adv_cosine, "comet": adv_comet}]
    base_acc = base_acc_num / len(prompts)
    adv_acc = adv_acc_num / len(prompts)
    data = [args.dataset_name, args.function_name, args.model_name, base_acc, adv_acc, base_score, adv_score, args.mask_ratio]

    with open(f"rebuttal_results/{args.model_name}/{args.output_dir}", "a") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(data) 

    print(f"write done")



                