
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import openai
import json
import os.path as osp
import asyncio
from asyncapi import process_api_requests_from_file
import torch
import logging
import ast
import matplotlib.pyplot as plt

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import re
import random
import numpy as np

system_role = """
You are an expert in analytical and logical reasoning. You will be given a main question and an image along with its full answer, 
and a subquestion and answer.
You task is to evaluate the correctness of the answer of the subquestion based on the main question, image and CoT.
"""

async def call_async_api(request_filepath, save_filepath, request_url, api_key, max_request_per_minute, max_tokens_per_minute, sp, ss):
    await process_api_requests_from_file(
            requests_filepath=request_filepath,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_request_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name='cl100k_base',
            max_attempts=int(2),
            logging_level=int(logging.INFO),
            seconds_to_pause=sp,
            seconds_to_sleep=ss
        )


def generate_chat_input_file(input_text, system_role, model_name = 'gpt-3.5-turbo', temperature = 0, n = 1):
    jobs = []
    for i, text in enumerate(input_text):
        obj = {}
        obj['model'] = model_name
        obj['messages'] = [
            {"role": "system", "content": system_role},
            {
                'role': 'user',
                'content': text 
            }
        ]
        obj['temperature'] = temperature
        obj['n'] = n
        jobs.append(obj)
    return jobs 

def openai_text_api(input_text, api_key, model_name = "gpt-3.5-turbo", temperature = 0, n = 1):
    response = openai.ChatCompletion.create(
        model=model_name,               
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": input_text},
        ],
        temperature=temperature,
        api_key=api_key,
        n = n)
    return response 


def efficient_openai_text_api(input_text, input_file, output_file, system_role, api_key="change_this_to_your_key", model_name="gpt-3.5-turbo", request_url = "https://api.openai.com/v1/chat/completions", rewrite = True, temperature = 0, n = 1):
    # import ipdb; ipdb.set_trace()
    # openai_result = []
    rewrite = True
    non_empty_results = []
    results = []
    if not osp.exists(output_file):
        jobs = generate_chat_input_file(input_text, system_role, model_name=model_name, temperature = temperature, n = n)

        with open(input_file, "w") as f:
            for i, job in enumerate(jobs):
                json_string = json.dumps(job)
                if job['messages'][0]['content'] != "":
                    f.write(json_string + "\n")
                    non_empty_results.append(i)
        asyncio.run(
            call_async_api(
                input_file, save_filepath=output_file,
                request_url=request_url,
                api_key=api_key,
                max_request_per_minute=60, 
                max_tokens_per_minute=10000,
                sp=60,
                ss=0
            )
        )
    openai_result = []

    with open(output_file, 'r') as f:
        # import ipdb; ipdb.set_trace()
        for i, line in enumerate(f):
            json_obj = json.loads(line.strip())
            content = json_obj[1]
            idx = json_obj[-1]
            choices = []
            non_empty_results.append(i)
            if content == "":
                openai_result.append(("", idx))
                # import ipdb; ipdb.set_trace()
            elif isinstance(idx, int):
                choices = [x['message']['content'] for x in json_obj[1]['choices']]
                openai_result.append((choices, idx))
            else:
                idx = json_obj[-2]
                new_result = openai_text_api(json_obj[0]['messages'][0]['content'], api_key, model_name = json_obj[0]['model'], temperature = json_obj[0]['temperature'], n = json_obj[0]['n'])
                choices = [x['message']['content'] for x in new_result['choices']]
                openai_result.append((choices, idx))

    openai_result = sorted(openai_result, key=lambda x:x[-1])
    results = [("", idx) for idx in range(len(input_text))]

    return openai_result