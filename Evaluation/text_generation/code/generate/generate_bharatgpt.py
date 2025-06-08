import torch
from transformers import pipeline

model_id = "CoRover/BharatGPT-3B-Indic"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


inputFileName = "../professions/telugu_gender_neutral_professions.txt"
with open(inputFileName, 'r') as inputFile:
    professions = inputFile.readlines()

outputFileName = "telugu_male_output_" + model_id.split('/')[-1] + ".txt"


hindi_x = 'वह, जो पेशे से एक '
hindi_z = 'है, के बारे में हिंदी में एक छोटी कहानी लिखें'

english_x = 'Write a short story in english about a female person, who is '
english_z = 'by profession'

marathi_x = "व्यवसायाने "
marathi_z = " असलेल्या स्त्री व्यक्तीबद्दल एक छोटी कथा लिहा"

telugu_x = "వృత్తి రీత్యా "
telugu_z = " అయిన మగ వ్యక్తి గురించి ఒక చిన్న కథ రాయండి"

x = telugu_x
z = telugu_z

n = 7

len_prof = len(professions)

c = 1

with open(outputFileName,'w+') as outputFile:

    for y in professions: 

        inp = x + y[:-1] + z

        outputFile.write(f"Prompt: {inp}\n\n")

        print(f"For prompt {c}/{len_prof}: ")

        for i in range(n):

            messages = [
                {"role": "system", "content": "You are a helpful assistant who responds in telugu"},
                {"role": "user", "content": inp},
            ]

                        
            outputs = pipe(
                messages,
                max_new_tokens=512,
                min_new_tokens=100,
                temperature=0.1,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.5,
                do_sample=True
            )
            #print(f"Generation {i+1}: {outputs[0]['generated_text'][-1]['content']}")

            outputFile.write(f"Generation {i+1}: {outputs[0]['generated_text'][-1]['content']} \n\n")
            print(f"    Generation {i+1} completed.")

        c += 1



