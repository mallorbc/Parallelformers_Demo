from transformers import AutoModelForCausalLM, AutoTokenizer
from parallelformers import parallelize

import argparse
import torch

import os
from parallelformers import parallelize

def fix_text(text):
    if text.find("&amp") != -1:
        text = text.replace("&amp","&")
    return text

clear = lambda: os.system('clear')

# model_name = "EleutherAI/gpt-j-6B"
model_name  = "EleutherAI/gpt-neo-2.7B"
# model_name  = "EleutherAI/gpt-neo-1.3B"

if __name__ == '__main__':
    parser = argparse.ArgumentParser("tool for running a pretrained gpt neo")
    parser.add_argument("-min","--min_output",default=10,type=int)
    parser.add_argument("-max","--max_output",default=50,type=int)
    parser.add_argument("-temp","--temp_output",default=0.8,type=float)
    parser.add_argument("-tk","--top_k",default=50,type=int)
    parser.add_argument("-tp","--top_p",default=0.95,type=float)
    parser.add_argument("-n","--num_output",default=1,type=int)

    args = parser.parse_args()
    default_min_len = args.min_output
    default_max_len = args.max_output
    default_temp = args.temp_output
    default_top_k = args.top_k
    default_top_p = args.top_p
    default_number_of_outputs = args.num_output


    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    parallelize(model, num_gpus=2, fp16=True)



    print("model loaded")

    while True:
        clear()
        prompt = str(input("What prompt do you want to give the AI? "))
        prompt = "<|endoftext|>" + prompt
        try: 
            min_output_length = int(input("Minimum output length? "))
        except:
            min_output_length = default_min_len
        try: 
            max_output_length = int(input("Maximum output length? "))
        except:
            max_output_length = default_max_len
        try:
            temp = float(input("What temperature to use? "))
        except:
            temp = default_temp
        try:
            top_k = int(input("What top k to use? "))
        except:
            top_k = default_top_k
        try:
            top_p = float(input("What top p to use? "))
        except:
            top_p = default_top_p
        try:
            number_of_outputs = int(input("Number of outputs? "))
        except:
            number_of_outputs = default_number_of_outputs



        encoding = tokenizer(prompt, return_tensors='pt')

        with torch.no_grad():
            generated_ids = model.generate(**encoding,do_sample=True, temperature=temp, max_length=max_output_length,min_length=min_output_length,num_return_sequences=number_of_outputs,top_k=top_k,top_p=top_p)
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
        for text in generated_texts:
            text = fix_text(text)
            print()
            print(text)

        delay=str(input())
        if delay == "0":
            break
