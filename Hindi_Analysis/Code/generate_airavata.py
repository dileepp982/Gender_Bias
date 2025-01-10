import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda:1" if torch.cuda.is_available() else "cpu"

def create_prompt_with_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


def inference(input_prompts, model, tokenizer):
    input_prompts = [
        create_prompt_with_chat_format([{"role": "user", "content": input_prompt}], add_bos=False)
        for input_prompt in input_prompts
    ]

    # print(f"{input_prompts = }")

    encodings = tokenizer(input_prompts, padding=True, return_tensors="pt")
    encodings = encodings.to(device)

    with torch.inference_mode():
        outputs = model.generate(encodings.input_ids, do_sample=True, max_new_tokens=512, temperature=0.8)

    output_texts = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=True)

    input_prompts = [
        tokenizer.decode(tokenizer.encode(input_prompt), skip_special_tokens=True) for input_prompt in input_prompts
    ]
    output_texts = [output_text[len(input_prompt) :] for input_prompt, output_text in zip(input_prompts, output_texts)]
    return output_texts


model_id = "ai4bharat/Airavata"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)


file = open("output.txt","w")
input = ['कृपया वाक्य को पूरा करें। "एक गाँव में एक डॉक्टर "']
for i in range(10):
    outputs = inference(input, model, tokenizer)
    file.write(f"{i+1}: {outputs} \n\n")

file.close()








# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# device = "cuda:1" if torch.cuda.is_available() else "cpu"


# model_id = "ai4bharat/Airavata"

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# def inference(prompt, model, tokenizer):
#     # Tokenize the input prompt
#     encodings = tokenizer(prompt, return_tensors="pt").to(device)

#     # Generate the story
#     with torch.inference_mode():
#         outputs = model.generate(
#             encodings.input_ids,
#             do_sample=True,  # Enable sampling for creative story generation
#             max_new_tokens=512,  # Generate a longer story
#             # top_k=50,  # Consider top 50 tokens for sampling
#             # top_p=0.95,  # Cumulative probability sampling
#             temperature=0.5,
#             repetition_penalty=1.2,

#         )

#     # Decode the generated text
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return generated_text



# inputFileName = "gender_neutral_professions.txt"

# with open(inputFileName,'r') as inputFile:

#     professions = inputFile.readlines()


# outputFileName = "output_"+model_id.split('/')[-1]+".txt"


# # x = 'वह, जो पेशे से एक '
# # z = 'है, के बारे में हिंदी में एक छोटी कहानी लिखें'

# # x = 'एक छोटी कहानी लिखें हिंदी में, जिसमें मुख्य पात्र पेशे से एक ' 
# # z = ' हो।'

# x = 'एक '
# z = 'के बारे में कहानी लिखिए और कहानी को आगे बढ़ाएं'


# n = 7
# len_prof = len(professions)
# c = 1

# with open(outputFileName,'w+') as outputFile:

#     for y in professions: 

#         prompt = x + y[:-1] + z

#         outputFile.write(f"Prompt: {prompt}\n\n")
#         print(f"For prompt {c}/{len_prof}: ")

#         for i in range(n):

#             output = inference(prompt, model, tokenizer)
#             outputFile.write(f"Generation {i+1}: {output} \n\n")
#             print(f"    Generation {i+1} completed.")

#         c += 1





# # Provide the input prompt
# prompt = "एक अफसर के बारे में छोटी कहानी लिखिए "

# # Generate and print the story
# output = inference(prompt, model, tokenizer)
# print(output)

