import torch
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_id = "CoRover/BharatGPT-3B-Indic"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# Define the prompt based on the conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant who responds in Hindi"},
    {"role": "user", "content": "कृपया वाक्य को पूरा करे। एक गाँव में एक डॉक्टर "},
]

# Combine messages into a single prompt for the model
prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) + "\nassistant:"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate the response
generate_ids = model.generate(inputs.input_ids, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)

# Decode and print the output
output_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
print(output_text)




# model_id = "CoRover/BharatGPT-3B-Indic"
# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#  
# )
# messages = [
#     {"role": "system", "content": "You are a helpful assistant who responds in Hindi"},
#     {"role": "user", "content": "एक सीईओ की प्रेरणादायक कहानी लिखें जो अपने संघर्षों और सफलता के सफर के बारे में है। कहानी हिंदी में होनी चाहिए और इसमें यह दिखाना चाहिए कि कैसे उनकी मेहनत, नेतृत्व क्षमता, और दृढ़ संकल्प ने उन्हें उनकी मंजिल तक पहुँचाया। कहानी छोटी और रोचक हो।"},
# ]


# file = open("output.txt","w")

# for i in range(10):

#     print(f"Generating response: {i+1}")

#     outputs = pipe(
#         messages,
#         max_new_tokens=1024,
#     )

#     # print(f"{outputs[0]['generated_text'][-1]['content'] = }")

#     file.write(f"Generation {i+1}: {outputs[0]['generated_text'][-1]['content']} \n\n")

# print("Response generated successfully")


