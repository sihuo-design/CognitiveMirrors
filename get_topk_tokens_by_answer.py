import argparse
from tqdm import tqdm
import json
import openai
import torch
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Specific pyvene imports
from utils import get_llama_activations_pyvene, tokenized_tqa
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv
import numpy as np
from llm import efficient_openai_text_api
import re

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

gpt_prompt = """
You are given a reference answer and a predicted answer to a sub-question.

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
3. If `"True"` is returned, extract and return the **5 most semantically important tokens** from the prediction answer.
   - These tokens should reflect the **core meaning** of the answer.
   - Avoid common stopwords unless they are crucial to the semantics.
4. If `"False"` is returned, write `"None"` for the tokens.

Output format:
[
{{
  "correct": <True|False>,
  "confidence": <float between 0 and 1>
  "top 5 tokens": <list of 5 tokens or "None">
}}
]
Your answer:
"""

gpt_prompt_onlyanswer = """
You are given two reference answers and a predicted answer to a question.

This is the reference answer1:
<reference_answer1>
{reference_answer1}
</reference_answer1>

This is the reference answer2:
<reference_answer2>
{reference_answer2}
</reference_answer2>

This is the predicted answer:
<prediction_answer>
{prediction_answer}
</prediction_answer>

Instructions:
1. Return `"True"` if the prediction answer is semantically consistent with the reference answer1. Otherwise, return `"False"`.
2. Reture `"True"` if the prediction answer is semantically consistent with the reference answer2. Otherwise, return `"False"`.

Output format:
[
{{
  "answer1": <True|False>,
  "answer2": <True|False>

}}
]
Your answer:
"""

def cot_prompt(dataset, llm_answer, args):
    all_prompts = []
    
    k = 0
    for i in range(len(dataset)):
        # question = dataset[i][0]['question']
        # answer = dataset[i][0]['answer']
        generated = dataset[i][0]['generated']

        for j in range(len(generated)):
            # subquestion = generated[j]['subquestion']
            ref_answer = generated[j]['answer']
            pre_answer = llm_answer[k][0]['answer']

            # format prompt
            if args.dataset_name == "extra_300":
                prompt_text = gpt_prompt_onlyanswer.format(
                    reference_answer=ref_answer,
                    prediction_answer=pre_answer
                )
            else:
                prompt_text = gpt_prompt.format(
                    reference_answer=ref_answer,
                    prediction_answer=pre_answer
                )
            k += 1

            all_prompts.append(prompt_text)

    return all_prompts

def cot_prompt_extra(dataset, llm_answer, args):
    all_prompts = []
    
    k = 0
    for i in range(len(dataset)):
        # question = dataset[i][0]['question']
        # answer = dataset[i][0]['answer']
        generated = dataset[i]['generated']
        ref_answer2 = generated[-1]['answer']
        ref_answer1 = dataset[i]['answer']

        pre_answer = llm_answer[i]

        # format prompt
        if args.dataset_name == "extra_300":
            prompt_text = gpt_prompt_onlyanswer.format(
                reference_answer1=ref_answer1,
                reference_answer2=ref_answer2,
                prediction_answer=pre_answer
            )

        k += 1

        all_prompts.append(prompt_text)

    return all_prompts

def safe_json_parse(json_text: str) -> dict:
    # check "answer" value and add quotes if missing
    def fix_answer_value(match):
        key, value = match.group(1), match.group(2).strip()
        if value.startswith('"') and value.endswith('"'):
            return f'{key}{value}'
        else:
            return f'{key}"{value}"'

    # match "answer"
    fixed_text = re.sub(r'("answer": )(.+?)(?=\s*})', fix_answer_value, json_text)

    return json.loads(fixed_text)

def get_result_from_output(outputs):
    answer_list = []
    confidence_list = []
    text_list = []
    
    for i, output in enumerate(outputs):
        # output = re.sub(r'"True"', 'True', output[0][0])
        # output = re.sub(r'"False"', 'False', output)
        output = output[0][0].replace("True", "true").replace("False", "false")
        matches = re.findall(r'\[\s*{[^}]+}\s*\]', output)
        if matches:
            output_json_block = matches[-1].strip()
            print("Extracted JSON answer block:", output_json_block)

            try:
                # parsed = safe_json_parse(output_json_block)
                parsed = json.loads(output_json_block)
                answer, confidence, text = parsed[0]["correct"], parsed[0]["confidence"], parsed[0]["top 5 tokens"]  # This will return '"$80"'
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
        text_list.append(text)
        
        # answer_list2.append(answer2)
        # confidence_list2.append(confidence2)
        
        print(f"Answer: {answer}, Confidence: {confidence}, Text: {text}")
    
    return answer_list, confidence_list, text_list


def get_result_from_output_extra(outputs):
    answer1_list = []
    answer2_list = []
    text_list = []
    
    for i, output in enumerate(outputs):
        # output = re.sub(r'"True"', 'True', output[0][0])
        # output = re.sub(r'"False"', 'False', output)
        output = output[0][0].replace("True", "true").replace("False", "false")
        matches = re.findall(r'\[\s*{[^}]+}\s*\]', output)
        if matches:
            output_json_block = matches[-1].strip()
            print("Extracted JSON answer block:", output_json_block)

            try:
                # parsed = safe_json_parse(output_json_block)
                parsed = json.loads(output_json_block)
                answer1, answer2 = parsed[0]["answer1"], parsed[0]["answer2"] # This will return '"$80"'
                print("Final extracted answer (with quotes):", answer1)
            except:
                answer1_match = re.search(r'"answer1":\s*"([^"]+)"', output)
                answer2_match = re.search(r'"answer2":\s*"([^"]+)"', output)
                # token_match = re.search(r'"top 5 tokens":\s*([\d.]+)', output)
                
                # 如果匹配到值则返回
                if answer1_match and answer2_match:
                    answer1 = answer1_match.group(1)
                    answer2 = answer2_match.group(1)
                    # text = token_match.group(1)

                else:
                    answer1 = ""
                    answer2 = ""
                    # text = ""
        
        answer1_list.append(answer1)
        answer2_list.append(answer2)
        # text_list.append(text)
        
        # answer_list2.append(answer2)
        # confidence_list2.append(confidence2)
        
        print(f"Answer1: {answer1}, Answer2: {answer2}")
    
    return answer1_list, answer2_list


system_role = """
You are an expert in analytical and logical reasoning. 
You task is to evaluate the correctness of the predicted answer.
"""

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1_8B_instruct')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='cot_qa')
    parser.add_argument('--key', default = 'your key', type = str)
    parser.add_argument('--temperature', default = 1, type = float)
    parser.add_argument('--llm_model_method', default = 'o4-mini', type = str)
    parser.add_argument('--mode', default = 'train', type = str)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    device = "cuda"
    dataset = json.load(open(f'dataset/balanced_cot_{args.mode}_data.json'))

    with open("head_results/output_" + args.model_name + "_" + args.dataset_name + "_" + args.mode + ".json", "r", encoding="utf-8") as file:
        llm_answer = json.load(file)
    
    print("Tokenizing prompts")
    
    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    all_prompts = cot_prompt(dataset, llm_answer, args)


    input_filename = "asy/input_answer_gpt_" + args.llm_model_method + args.model_name + "_" + args.dataset_name + "_" + args.mode + ".json"
    output_filename = "asy/output_answer_gpt_" + args.llm_model_method + args.model_name + "_" + args.dataset_name + "_" + args.mode + ".json"
    outputs = efficient_openai_text_api(all_prompts, input_filename, output_filename, system_role=system_role, model_name=args.llm_model_method, api_key=args.key, temperature=args.temperature, n=1)
    
    answer_list, confidence_list, text_list = get_result_from_output(outputs)
    with open(f'head_results/{args.model_name}_{args.dataset_name}_token_positions_{args.mode}.pkl', "rb") as f:
        meaning_token_positions_list = pickle.load(f)
    with open("head_results/output_" + args.model_name + "_" + args.dataset_name + "_" + args.mode + "_with_gpt_label.json", "r") as f:
        llm_answer = json.load(f)
    
    topk_position_list = []    
    for i in range(0, len(meaning_token_positions_list)):
        #print("meaning_token_positions_list[i]: ", meaning_token_positions_list[i])
        #print(i)
        if len(meaning_token_positions_list[i]) < 6:
            topk_position_list.append(meaning_token_positions_list[i])
            continue
        
        meaning_token_positions = meaning_token_positions_list[i]
        true_answer = llm_answer[i][0]['answer']
        true_answer_idx = tokenizer.encode(true_answer, add_special_tokens=False)
        topk_answer = text_list[i]

        if topk_answer == "None":
            topk_position_list.append(meaning_token_positions_list[i])
            continue
        topk_position = []
        #print(i)
        
        # if "" in topk_answer:
        #     topk_answer.remove('')
        topk_answer = [token for token in topk_answer if token]
        for token in topk_answer:
            idx = []
            #print(token)
            idx.append(tokenizer.encode(token, add_special_tokens=False))
            idx.append(tokenizer.encode(" " + token, add_special_tokens=False))
            #print(idx)
            for id in idx:
                if id[0] in true_answer_idx:
                    pos = true_answer_idx.index(id[0])
                    break
                else:
                    pos = -1  #set -1 if not found
            topk_position.append(meaning_token_positions[pos])     ###all token   
        topk_position_list.append(topk_position)    

    ###save topk token postion 
    with open(f'{args.model_name}_{args.dataset_name}_topk_position_{args.mode}.pkl', "wb") as f:
        pickle.dump(topk_position_list, f)
    
    
if __name__ == '__main__':
    main()