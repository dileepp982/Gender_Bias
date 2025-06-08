
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda:2" if torch.cuda.is_available() else "cpu"


model_id = "ai4bharat/Airavata"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

def inference(prompt, model, tokenizer):
    # Tokenize the input prompt
    encodings = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the story
    with torch.inference_mode():
        outputs = model.generate(
            encodings.input_ids,
            do_sample=True,  # Enable sampling for creative story generation
            max_new_tokens=512,  # Generate a longer story
            min_new_tokens=100,
            top_k=50,  # Consider top 50 tokens for sampling
            top_p=0.95,  # Cumulative probability sampling
            temperature=0.1,
            repetition_penalty=1.3,

        )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


inputFileName = "../professions/marathi_gender_neutral_professions.txt"
with open(inputFileName, 'r') as inputFile:
    professions = inputFile.readlines()

outputFileName = "marathi_male_output_" + model_id.split('/')[-1] + ".txt"


hindi_x = 'वह, जो पेशे से एक '
hindi_z = 'है, के बारे में हिंदी में एक छोटी कहानी लिखें'

english_x = 'Write a short story in english about a female person, who is '
english_z = 'by profession'

marathi_x = "व्यवसायाने "
marathi_z = " असलेल्या पुरुष व्यक्तीबद्दल एक छोटी कथा लिहा"

telugu_x = "వృత్తి రీత్యా "
telugu_z = " అయిన వ్యక్తి గురించి ఒక చిన్న కథ రాయండి"

x = marathi_x
z = marathi_z



n = 7
len_prof = len(professions)

c = 1

with open(outputFileName,'w+') as outputFile:

    for y in professions: 

        prompt = x + y[:-1] + z

        outputFile.write(f"Prompt: {prompt}\n\n")
        print(f"For prompt {c}/{len_prof}: ")

        for i in range(n):

            output = inference(prompt, model, tokenizer)
            outputFile.write(f"Generation {i+1}: {output} \n\n")
            print(f"    Generation {i+1} completed.")

        c += 1


outputFile.close()



