#!/opt/software/install/miniconda37/bin/python
import argparse
parser = argparse.ArgumentParser(description='Running a chatbot with gradio')
parser.add_argument('--model_path', type=str, help='model path, for example `llama/7b-32`')
parser.add_argument('--input_file', type=str, help='input file')
parser.add_argument('--device', type=str, help='device (default: cuda:0)', default='cuda:0')
args = parser.parse_args()

model_name = args.model_path.replace("/", "_") # llama/7b-32 --> llama_7b-32
offset = -4
if 'llama' in model_name:
    offset = -7

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import json
import pandas as pd
from tqdm import tqdm
import sys

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path)
model = model.half()
model = model.to(args.device) #, device_map='auto')
#model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='balanced')
model.eval()

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def generate_response(instruction: str, input_text: str, **kwargs) -> str:
    #input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to(device)
    input_ids = tokenizer(PROMPT_DICT['prompt_input'].format(instruction=instruction, input=input_text), return_tensors="pt").input_ids.to(args.device)

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=False, max_new_tokens=2048, top_p=0.92, top_k=0, **kwargs)[0].cpu()

    s = tokenizer.decode(gen_tokens)

    ss = s.split("### Response:")[1].strip()[0:offset]
    return ss

#with open('data/test_data_points-v2-128.json') as f:
with open(args.input_file) as f:
    d = json.load(f)

#x = pd.read_csv('data/test_data_points.csv.gz')
#assert len(d) == len(x)

results = []
for a in tqdm(d):
    response = generate_response(a['instruction'], a['input'])
    a['response'] = response
    results.append(a)

#with open('data/test_data_points-v2-7b-128-predictions.json', 'w') as f:
outfile = args.input_file.replace('.json', f'-{model_name}-predictions.json')
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)

