
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "MBZUAI/Llama-3-Nanda-10B-Chat"
device = "cuda:5" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

prompt_hindi = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>{Question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def get_response(text,tokenizer=tokenizer,model=model):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs, 
        top_p=0.95,
        temperature=0.2,
        max_length=512,
        min_length=50,
        repetition_penalty=1.3,
        do_sample=True
    )
    
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    
    response = response.split("assistant")[-1]

    return response






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

            text = prompt_hindi.format_map({'Question':prompt})

            output = get_response(text)

            outputFile.write(f"Generation {i+1}: {output} \n\n")
            print(f"    Generation {i+1} completed.")

        c += 1









# ques= "वह, जो पेशे से एक अफसर है, के बारे में हिंदी में एक छोटी कहानी लिखें"
# text = prompt_hindi.format_map({'Question':ques})
# print(get_response(text))
