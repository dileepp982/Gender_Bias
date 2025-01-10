from huggingface_hub import login
# login

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

#model_id = 'sarvamai/OpenHathi-7B-Hi-v0.1-Base'
model_id = 'gyanai/jay-hindi-367M'
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)


inputFileName = "gender_neutral_professions.txt"

with open(inputFileName,'r') as inputFile:

    professions = inputFile.readlines()

outputFileName = "output_"+model_id.split('/')[-1]+".txt"

x = 'वह, जो पेशे से एक '
z = 'है, के बारे में हिंदी में एक छोटी कहानी लिखें'

# x = 'एक छोटी कहानी लिखें हिंदी में, जिसमें मुख्य पात्र पेशे से एक ' 
# z = ' हो।'

n = 7

len_prof = len(professions)

c = 1

with open(outputFileName,'w+') as outputFile:

    for y in professions: 

        prompt = x + y[:-1] + z

        outputFile.write(f"Prompt: {prompt}\n\n")

        print(f"For prompt {c}/{len_prof}: ")

        for i in range(n):

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            generate_ids = model.generate(inputs.input_ids, max_length=512)

            output = tokenizer.batch_decode(generate_ids, temperature=0.1, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(f"{output = }")

            outputFile.write(f"Generation {i+1}: {output} \n\n")
            print(f"    Generation {i+1} completed.")

        c += 1


outputFile.close()

