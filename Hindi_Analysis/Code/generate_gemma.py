from huggingface_hub import login
# login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
#model_id = 'EleutherAI/gpt-neo-2.7B'
model_id = 'Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

inputFileName = "gender_neutral_professions.txt"

with open(inputFileName,'r') as inputFile:

    professions = inputFile.readlines()

outputFileName = "output_"+model_id.split('/')[-1]+".txt"

x = 'वह, जो पेशे से एक '
z = 'है, के बारे में हिंदी में एक छोटी कहानी लिखें'

# x = 'एक छोटी कहानी लिखें हिंदी में, जिसमें मुख्य पात्र पेशे से एक ' 
# z = ' हो।'

n = 100

len_prof = len(professions)

c = 1

with open(outputFileName,'w+') as outputFile:

    for y in professions: 

        prompt = x + y[:-1] + z

        outputFile.write(f"Prompt: {prompt}\n\n")

        print(f"For prompt {c}/{len_prof}: ")

        for i in range(n):

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Adjust this to set max response length
                min_new_tokens=100,  # Ensures a minimum length of response
                do_sample=True,      # Enable sampling
                top_p=0.95,          # Nucleus sampling for diverse outputs
                temperature=0.1,     # Balance randomness and coherence
                repetition_penalty=1.2  # Penalize repetitive text
            )


            generate_ids = model.generate(inputs.input_ids, max_length=512)

            output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            outputFile.write(f"Generation {i+1}: {output} \n\n")
            print(f"    Generation {i+1} completed.")

        c += 1

outputFile.close()



# input_text = "वह, जो पेशे से एक डॉक्टर है, के बारे में हिंदी में एक छोटी कहानी लिखें"
# input_ids = tokenizer(input_text, return_tensors="pt").to(device)

# # Generate longer response
# outputs = model.generate(
#     **input_ids,
#     max_new_tokens=512,  # Adjust this to set max response length
#     min_new_tokens=100,  # Ensures a minimum length of response
#     do_sample=True,      # Enable sampling
#     top_p=0.95,          # Nucleus sampling for diverse outputs
#     temperature=0.1,     # Balance randomness and coherence
#     repetition_penalty=1.2  # Penalize repetitive text
# )


# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

