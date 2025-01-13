import numpy as np
import pandas as pd
import re
import ast
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub.hf_api import HfFolder

class Load_Model:
    def __init__(self, model_name):
        self.model_name = model_name
    def __call__(self):
        # Import the necessary library
        from huggingface_hub.hf_api import HfFolder

        # Save the Hugging Face API token
        # HfFolder.save_token("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")
        token = "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"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left',token = token)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, token = token)
        tokenizer.pad_token = tokenizer.bos_token
        return model, tokenizer
