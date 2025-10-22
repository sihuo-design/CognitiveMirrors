import argparse
from tqdm import tqdm
import json
import torch
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import openai
# Specific pyvene imports
from utils import get_llama_activations_pyvene, tokenized_tqa, cot_prompt, get_qwen_activations_pyvene
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv
import numpy as np

HF_NAMES = {
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


system_role = """
You are an expert in analytical and logical reasoning. You will be given a main question, prior knowledge in chain-of-thought (CoT) format, 
and a follow-up subquestion and answer.
You task is to evaluate the correctness of the answer of the follow-up subquestion based on the main question and CoT.
"""

gpt_prompt = """
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

Here is the answer of the subquestion:
<subquestion_answer>
{answer}
</subquestion_answer>

Instructions:
- Evaluate the correctness of the answer to the subquestion. Respond with True if the answer is factually and logically correct. Respond with False if the answer is incorrect or incomplete.

- Provide a confidence score between 0 and 1, where: 1 indicates full confidence in your judgment. Values closer to 0 indicate lower confidence.

Output format:
[
{
  "correct": <True|False>,
  "confidence": <float between 0 and 1>
}
]

Your answer:
"""

def nocot_prompt(dataset, args, tokenizer):
    system_role = """
    You are an expert in analytical and logical reasoning. Your task is to answer the question.
    """

    prompt = """
    {system_role}
    Here is the question:
    <question>
    {question}
    </question>
    
    Only output your final answer using this format:
    [
    {{
        "answer": "<Your final answer here>"
    }}
    ]

    Your answer:
    """
    all_prompts = []
    all_labels = []
    all_answers = []
    for i in range(len(dataset)):
        question = dataset[i]['question']
        answer = dataset[i]['answer']
        if "qwen" in args.model_name: 
            text = prompt.format(
                        system_role=system_role,
                        question=question
                    )
            messages = [
                            {"role": "user", "content": text}
                        ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
            )
        else:
            text = prompt.format(
                    system_role=system_role,
                    question=question
                    )
        all_prompts.append(text)
        all_answers.append(answer)

    return all_prompts, all_labels, all_answers       

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.2_3B_instruct')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='cot_qa')
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--use_setoken', default=False, action='store_true')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
   
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    device = "cuda"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.dataset_name == "cot_qa":
        dataset = json.load(open(f'dataset/balanced_cot_{args.mode}_data.json'))
        formatter = tokenized_tqa
    elif args.dataset_name == "extra_300":
        dataset = json.load(open(f'dataset/extra_300.json'))
        formatter = tokenized_tqa
    else: 
        raise ValueError("Invalid dataset name")
      
    print("Tokenizing prompts")
    if args.dataset_name == "extra_300":
        prompts, labels, answers = nocot_prompt(dataset, args, tokenizer)
    else:        
        prompts, labels, answers = cot_prompt(dataset)

    collectors = []
    pv_config = []
    if "gemma" in args.model_name:
        for layer in range(model.config.text_config.num_hidden_layers): 
            collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
            collectors.append(collector)
            pv_config.append({
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(collector),
            })
    else:
        for layer in range(model.config.num_hidden_layers): 
            collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
            collectors.append(collector)
            pv_config.append({
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(collector),
            })
    collected_model = pv.IntervenableModel(pv_config, model)

    all_layer_wise_activations = []
    all_head_wise_activations = [] 
    output_answers = []
    output_llm = []
    token_positions_list = []
    print("Getting activations")
    i = 0
    save_labels = []
    for i, prompt in tqdm(enumerate(prompts)):
        
        head_wise_activations, _, output_answer, token_positions = get_llama_activations_pyvene(tokenizer, collected_model, collectors, prompt, device, args)

        all_head_wise_activations.append(head_wise_activations.copy())
        output_answers.append(output_answer)
        output_llm.append([{"prompt": prompt, "truth": answers[i], "answer": output_answer}])
    
        token_positions_list.append(token_positions)
        i = i + 1

    with open("output_" + args.model_name + "_" + args.dataset_name + "_" + args.mode + ".json", "w") as f:
        json.dump(output_llm, f, indent=4)
        
    print("Saving")    
    with open(f'{args.model_name}_{args.dataset_name}_token_positions_{args.mode}.pkl', 'wb') as f:
        pickle.dump(token_positions_list, f)

    
    print("Saving head wise activations")    
    with open(f'{args.model_name}_{args.dataset_name}_head_wise_{args.mode}.pkl', 'wb') as f:
        pickle.dump(all_head_wise_activations, f)


if __name__ == '__main__':
    main()
