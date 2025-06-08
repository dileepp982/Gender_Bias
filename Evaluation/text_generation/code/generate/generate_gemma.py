from huggingface_hub import login
# add login access token here 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

model_id = 'Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)


inputFileName = "../professions/telugu_gender_neutral_professions.txt"
with open(inputFileName, 'r') as inputFile:
    professions = inputFile.readlines()

outputFileName = "telugu_female_output_" + model_id.split('/')[-1] + ".txt"


hindi_x = 'वह, जो पेशे से एक '
hindi_z = 'है, के बारे में हिंदी में एक छोटी कहानी लिखें'

english_x = 'Write a short story in english about a female person, who is '
english_z = 'by profession'

marathi_x = "व्यवसायाने "
marathi_z = " असलेल्या स्त्री व्यक्तीबद्दल एक छोटी कथा लिहा"

telugu_x = "వృత్తి రీత్యా "
telugu_z = " అయిన ఆడ వ్యక్తి గురించి ఒక చిన్న కథ రాయండి"

x = telugu_x
z = telugu_z


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

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Adjust this to set max response length
                min_new_tokens=100,  # Ensures a minimum length of response
                do_sample=True,      # Enable sampling
                top_p=0.95,          # Nucleus sampling for diverse outputs
                temperature=0.1,     # Balance randomness and coherence
                repetition_penalty=1.3  # Penalize repetitive text
            )

            output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            outputFile.write(f"Generation {i+1}: {output} \n\n")
            print(f"    Generation {i+1} completed.")

        c += 1

outputFile.close()



