# Utils to work with pyvene

import os
import sys
sys.path.insert(0, "TruthfulQA")
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from tqdm import tqdm
import numpy as np

import pandas as pd
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle
from functools import partial
import random
from torch.cuda.amp import autocast
import time

import openai
import re
import json
import evaluate

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import OpenAI
client = OpenAI(api_key="key")

openai.api_key = "key"

ENGINE_MAP = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
}



def get_llama_activations_bau(model, prompt, device): 
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS+MLPS) as ret:
        # with TraceDict(model, HEADS+MLPS, retain_input=True) as ret:
            output = model(prompt, output_hidden_states = True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states


def find_subsequence(subseq_o, seq):
    index = -1
    """Return the start index of subseq in seq, or -1 if not found"""
    if len(subseq_o) > 1:
        subseq = subseq_o[:-1]  
    else:
        subseq = subseq_o
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i:i+len(subseq)] == subseq:
            index = i
            return i
    if index == -1:
        subseq = subseq_o[1:] 
        for i in range(len(seq) - len(subseq) + 1):
            if seq[i:i+len(subseq)] == subseq:
                index = i -1
                return i-1
    if index == -1:
        subseq = subseq_o[1:-1] 
        for i in range(len(seq) - len(subseq) + 1):
            if seq[i:i+len(subseq)] == subseq:
                index = i -1
                return i-1
    return -1

class StopOnCloseBracket(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_token_id = tokenizer.convert_tokens_to_ids("]")

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id

def safe_json_parse(json_text: str) -> dict:

    def fix_answer_value(match):
        key, value = match.group(1), match.group(2).strip()

        if value.startswith('"') and value.endswith('"'):
            return f'{key}{value}'
        else:
            return f'{key}"{value}"'

    fixed_text = re.sub(r'("answer": )(.+?)(?=\s*})', fix_answer_value, json_text)

    return json.loads(fixed_text)


def get_qwen_activations_pyvene(tokenizer, collected_model, collectors, prompt, device, args):
    output_answers = []
    no_output = []
    stopping_criteria = StoppingCriteriaList([StopOnCloseBracket(tokenizer)])
    with torch.no_grad():

        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = collected_model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = tokenizer.decode(output_ids[0:], skip_special_tokens=True).strip("\n")

        matches = re.findall(r'\[\s*{[^}]+}\s*\]', content)
        if matches:
            output_json_block = matches[-1].strip()
            print("Extracted JSON answer block:", output_json_block)

            try:
                parsed = safe_json_parse(output_json_block)

                if isinstance(parsed, list) and isinstance(parsed[0], dict) and "answer" in parsed[0]:
                    output_answer = parsed[0]["answer"]
                else:
                    output_answer = None 
                print("Final extracted answer (with quotes):", output_answer)
            except json.JSONDecodeError:
                output_answer = None
                print("Failed to decode JSON.")
        else:
            print("No JSON block found.")

        print("Generated output:", output_text)

    generated_ids = base_generated[0]  # [seq_len]

    generated_only_ids = generated_ids[prompt["input_ids"].shape[1]:].tolist()
    if output_answer is None:
        output_answer = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
            
    try:
        output_answer_str = str(output_answer).strip()
        output_text_ids = tokenizer.encode(output_answer_str, add_special_tokens=False)
    except Exception as e:
        print(f"Encoding failed for output_answer={output_answer}. Error: {e}")
        output_text_ids = []

    relative_start = find_subsequence(output_text_ids, generated_only_ids)
    if relative_start != -1:
        # absolute_start = prompt["input_ids"].shape[1] + relative_start
        absolute_start = relative_start
        token_positions = list(range(absolute_start, absolute_start + len(output_text_ids)))
        print("Token positions for output_text:", token_positions)
    else:
        print("output_text tokens not found in generated sequence.")
        token_positions = list(range(len(generated_only_ids)-1))
     
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            # collector.states: [num_tokens, num_heads, dim]
            states_per_gen = torch.stack(collector.states, axis=0)  # shape: [T, H, D]
            if args.use_setoken:
                selected_states = states_per_gen[token_positions]      # shape: [5, H, D]
            else:
                selected_states = states_per_gen
            head_wise_hidden_states.append(selected_states.cpu().numpy())
        else:
            head_wise_hidden_states.append(None)
        collector.reset()

    mlp_wise_hidden_states = []  
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).numpy()
    return head_wise_hidden_states, mlp_wise_hidden_states, output_answer, token_positions

def get_llama_activations_pyvene(tokenizer, collected_model, collectors, prompt, device, args):
    output_answers = []
    no_output = []
    stopping_criteria = StoppingCriteriaList([StopOnCloseBracket(tokenizer)])
    with torch.no_grad():
        # prompt = "What is the capital of France?"
        prompt = tokenizer(prompt, return_tensors='pt')        
        prompt = prompt.to(device)
        # output = collected_model({"input_ids": prompt.input_ids, "output_hidden_states": True})[1]
        prompt = {k: v.to(device) for k, v in prompt.items()}

        base_generated, adv_generated = collected_model.generate(
            base=prompt,
            max_new_tokens=128,
            do_sample=False,
            output_original_output=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,

        )
        output_text = tokenizer.decode(base_generated[0], skip_special_tokens=True)
        matches = re.findall(r'\[\s*{[^}]+}\s*\]', output_text)
        if matches:
            output_json_block = matches[-1].strip()
            print("Extracted JSON answer block:", output_json_block)

            try:
                parsed = safe_json_parse(output_json_block)
                if isinstance(parsed, list) and isinstance(parsed[0], dict) and "answer" in parsed[0]:
                    output_answer = parsed[0]["answer"]
                else:
                    output_answer = None 
                print("Final extracted answer (with quotes):", output_answer)
            except json.JSONDecodeError:
                output_answer = None
                print("Failed to decode JSON.")
        else:
            print("No JSON block found.")
        print("Generated output:", output_text)

    generated_ids = base_generated[0]  # [seq_len]

    generated_only_ids = generated_ids[prompt["input_ids"].shape[1]:].tolist()
    if output_answer is None:
        output_answer = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
            
    try:
        output_answer_str = str(output_answer).strip()
        output_text_ids = tokenizer.encode(output_answer_str, add_special_tokens=False)
    except Exception as e:
        print(f"Encoding failed for output_answer={output_answer}. Error: {e}")
        output_text_ids = []

    relative_start = find_subsequence(output_text_ids, generated_only_ids)
    if relative_start != -1:
        # absolute_start = prompt["input_ids"].shape[1] + relative_start
        absolute_start = relative_start
        token_positions = list(range(absolute_start, absolute_start + len(output_text_ids)))
        print("Token positions for output_text:", token_positions)
    else:
        print("output_text tokens not found in generated sequence.")
        token_positions = list(range(len(generated_only_ids)-1))

    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            # collector.states: [num_tokens, num_heads, dim]
            states_per_gen = torch.stack(collector.states, axis=0)  # shape: [T, H, D]
            if args.use_setoken:
                selected_states = states_per_gen[token_positions]      # shape: [5, H, D]
            else:
                selected_states = states_per_gen
            head_wise_hidden_states.append(selected_states.cpu().numpy())
        else:
            head_wise_hidden_states.append(None)
        collector.reset()

    mlp_wise_hidden_states = []  
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).numpy()
    return head_wise_hidden_states, mlp_wise_hidden_states, output_answer, token_positions


def get_llama_logits(model, prompt, device): 

    model.eval()
    with torch.no_grad(): 
        prompt = prompt.to(device)
        logits = model(prompt).logits
        logits = logits.detach().cpu()
        return logits

system_role = """
You are an expert in analytical and logical reasoning. You will be given a main question and prior knowledge in chain-of-thought (CoT) format.
Your task is to answer a follow-up subquestion using the information provided.
"""

prompt = """
{system_role}
Here is the main question:
<main_question>
{question}
</main_question>

Here is the prior knowledge in chain-of-thought (CoT) format:
<prior_knowledge>
{cot}
</prior_knowledge>

Here is the subquestion:
<subquestion>
{subquestion}
</subquestion>

Instructions:
- Answer the subquestion carefully.
- You can use the information in the prior_knowledge to help you answer the subquestion.
- Your response should be clear and concise.
- Stick to factual reasoning based on provided CoT.
- Do not include any explanation, commentary, or code.
- Do not output anything after the closing square bracket `]`.

Only output your final answer using this format:
[
    {{"answer": "<Your answer here>"}}
]

Your answer:
"""

def cot_prompt(dataset, low_function_name=None, high_function_name=None):
    all_prompts = []
    all_labels = []
    all_answers = []


    # define logic type mapping
    infor_extract = ["Retrieval", "Knowledge Recall", "Semantic Understanding", "Syntactic Understanding"]
    higher_logic = ["Induction", "Inference", "Logical Reasoning", "Decision-making"]

    for i in range(len(dataset)):
        question = dataset[i]['question']
        generated = dataset[i]['generated']

        for j in range(len(generated)):
            subquestion = generated[j]['subquestion']
            cot = ""

            # Accumulate all previous sub-QA pairs for context
            if j > 0:                
                for k in range(j):
                    if generated[k]["cognitive_skill"] != "Retrieval":
                        continue
                    prev_subq = generated[k]['subquestion']
                    prev_ans = generated[k]['answer']
                    cot += f"Q{k+1}: {prev_subq}\nA{k+1}: {prev_ans}\n"
            else:
                cot += "No prior knowledge.\n"

            # label: 0 for lower-level (information extraction), 1 for higher-level reasoning
            cognitive_skill = generated[j]['cognitive_skill']
            # label = 0 if cognitive_skill in infor_extract else 1
            label = cognitive_skill
            
            answer = generated[j]['answer']

            # format prompt
            prompt_text = prompt.format(
                system_role=system_role,
                question=question,
                cot=cot.strip(),
                subquestion=subquestion
            )

            all_prompts.append(prompt_text)
            all_labels.append(label)
            all_answers.append(answer)

    return all_prompts, all_labels, all_answers

def get_filtered_data(dataset, model, tokenizer):
    prompts, labels = cot_prompt(dataset)
    responses = []
    for i, input_text in enumerate(prompts):

        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {key: tensor.cuda() for key, tensor in inputs.items()}  # Move each tensor to CUDA

        # Generate text
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.0, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )

        # Decode generated text
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Print the generated text
        print("Generated Text:\n", decoded)
        answer_start = decoded.find("Your answer:")
        if answer_start != -1:
            response_text = decoded[answer_start + len("Your answer:"):].strip()
        else:
            response_text = decoded.strip()

        responses.append(response_text)    
    # Filter out responses that are not in the expected format
    filtered_responses = []
    
    return filtered_responses, labels  

def adv_generate(collected_model, tokenizer, prompt, device):
    stopping_criteria = StoppingCriteriaList([StopOnCloseBracket(tokenizer)])
    with torch.no_grad():
        base_generated, adv_generated = collected_model.generate(
            base=prompt,
            max_new_tokens=128,
            do_sample=False,
            output_original_output=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )

    return base_generated, adv_generated

def get_answer(base_generated, tokenizer): 
     
    output_text = tokenizer.decode(base_generated[0], skip_special_tokens=True)
    matches = re.findall(r'\[\s*{[^}]+}\s*\]', output_text)
    output_answer = None
    if matches:
        output_json_block = matches[-1].strip()

        try:
            parsed = safe_json_parse(output_json_block)
            if isinstance(parsed, list) and isinstance(parsed[0], dict) and "answer" in parsed[0]:
                output_answer = parsed[0]["answer"]
            else:
                output_answer = None 
            print("Final extracted answer:", output_answer)
        except:
            print("Failed to decode JSON.")
    else:
        print("No JSON block found.")
        
    return output_answer


bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge", experiment_id=f"comet_{os.getpid()}_{int(time.time()*1000)}")
comet = evaluate.load("comet", experiment_id=f"comet_{os.getpid()}_{int(time.time()*1000)}")

def emb_similarity(texts):
    texts = [each[0] for each in texts]
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    embeddings = [np.array(e.embedding) for e in response.data]

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarity = cosine_similarity(embeddings[0], embeddings[1])
    return similarity

def evaluate_metrics(predictions, references, predictions_full, references_full, sources=None, args=None, evaluate_model=None):
    predictions = [p if p.strip() != "" else "N/A" for p in predictions]
    references = [r if r.strip() != "" else "N/A" for r in references]
    results = {}

    # BLEU

    bleu_score = bleu.compute(predictions=predictions_full, references=references_full)
    results["bleu"] = bleu_score["bleu"]

    # ROUGE
    rouge_score = rouge.compute(predictions=predictions, references=sources)
    results.update(rouge_score)  # rouge1, rouge2, rougeL, rougeLsum
    
    cosine_score = emb_similarity([predictions, sources])
    results["cosine"] = cosine_score

    # COMET
    if args.use_comet and sources is not None:
        comet_score = comet.compute(
            predictions=predictions,
            references=references,
            sources=sources
        )
        results["comet"] = comet_score["mean_score"]
    else:
        results["comet"] = "Skipped (no sources provided)"

    return results

def simi_scoring(prompt, label, source_answer, model, tokenizer, intervened_model, device, args, evaluate_model):
    prompt = tokenizer(prompt, return_tensors='pt')    
    prompt = prompt.to(device)

    prompt = {k: v.cuda() for k, v in prompt.items()}

    base_generated, adv_generated = adv_generate(intervened_model, tokenizer, prompt, device)
    base_tokens = base_generated.sequences
    adv_tokens = adv_generated.sequences
        
    base_answer = get_answer(base_tokens, tokenizer)
    adv_answer = get_answer(adv_tokens, tokenizer)

    if args.use_bleu:
        base_generated_only_ids = base_tokens[0][prompt["input_ids"].shape[1]:].tolist()
        # adv_generated_ids = adv_tokens[0]
        adv_generated_only_ids = adv_tokens[0][prompt["input_ids"].shape[1]:].tolist()
        base_answer_full = tokenizer.decode(base_generated_only_ids, skip_special_tokens=True)
    # if adv_answer is None:
        adv_answer_full = tokenizer.decode(adv_generated_only_ids, skip_special_tokens=True)   
    if base_answer is None or adv_answer is None:
        base_answer, adv_answer = base_answer_full, adv_answer_full

    scores = evaluate_metrics([str(adv_answer)], [str(base_answer)], [str(adv_answer_full)], [str(base_answer_full)], sources=[str(source_answer)], args=args, evaluate_model=evaluate_model)  

    if "kl_scores" not in locals():
        kl_scores = []
    return base_answer, adv_answer, scores, kl_scores


def get_intervation_result(prompt, tokenizer, intervened_model, device, args):
    prompt = tokenizer(prompt, return_tensors='pt')    
    prompt = prompt.to(device)
    # output = collected_model({"input_ids": prompt.input_ids, "output_hidden_states": True})[1]
    prompt = {k: v.cuda() for k, v in prompt.items()}

    # base_generated, adv_generated = adv_generate(intervened_model, tokenizer, prompt, device)
    stopping_criteria = StoppingCriteriaList([StopOnCloseBracket(tokenizer)])
    with torch.no_grad():
        base_generated, adv_generated = intervened_model.generate(
            base=prompt,
            max_new_tokens=1024,
            do_sample=False,
            output_original_output=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id
        )

    base_tokens = base_generated
    adv_tokens = adv_generated

    base_generated_only_ids = base_tokens[0][prompt["input_ids"].shape[1]:].tolist()
    adv_generated_only_ids = adv_tokens[0][prompt["input_ids"].shape[1]:].tolist()
    base_answer_full = tokenizer.decode(base_generated_only_ids, skip_special_tokens=True)
    adv_answer_full = tokenizer.decode(adv_generated_only_ids, skip_special_tokens=True)   
    
    print("Base answer:", base_answer_full, "Adv answer:", adv_answer_full)
    return base_answer_full, adv_answer_full


def get_separated_activations(labels, head_wise_activations): 

    # separate activations by question
    dataset=load_dataset('truthful_qa', 'multiple_choice')['validation']
    actual_labels = []
    for i in range(len(dataset)):
        actual_labels.append(dataset[i]['mc2_targets']['labels'])

    idxs_to_split_at = np.cumsum([len(x) for x in actual_labels])        

    labels = list(labels)
    separated_labels = []
    for i in range(len(idxs_to_split_at)):
        if i == 0:
            separated_labels.append(labels[:idxs_to_split_at[i]])
        else:
            separated_labels.append(labels[idxs_to_split_at[i-1]:idxs_to_split_at[i]])
    assert separated_labels == actual_labels

    separated_head_wise_activations = np.split(head_wise_activations, idxs_to_split_at)

    return separated_head_wise_activations, separated_labels, idxs_to_split_at

def get_com_directions(num_layers, num_heads, usable_idxs, head_wise_activations, answers): 
    labels = []
    for i in usable_idxs:
        labels.append(answers[i][0]['label'])
    str_indices = [i for i, v in enumerate(labels) if isinstance(v, str)]
    for i in str_indices:
        if labels[i] == "True" or labels[i] == "true":
            labels[i] = True
        elif labels[i] == "False" or labels[i] == "false":
            labels[i] = False

    usable_labels = [int(l) for l in labels]
    
    com_directions = []
  
    for layer in tqdm(range(num_layers), desc="get_com_directions"): 
        for head in range(num_heads): 
            usable_head_wise_activations = np.concatenate([head_wise_activations[i].reshape(num_layers, num_heads, -1)[layer,head,:] for i in usable_idxs], axis=0)
            true_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)

    return com_directions