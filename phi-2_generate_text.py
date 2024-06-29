from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
model_name = "microsoft/phi-2"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
model_name = "microsoft/phi-2"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model and tokenizer loaded successfully.")

# Define your input text
input_text = "Once upon a time"
print(f"Input text: {input_text}")

# Encode the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
print(f"Encoded input ids: {input_ids}")
print(f"Attention mask: {attention_mask}")

# Generate text
output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)
print(f"Generated output ids: {output}")

# Decode the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Output text: {output_text}")