import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Load the model and tokenizer
model_name = "microsoft/phi-2"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model and tokenizer loaded successfully.")

# Load the MMLU dataset
dataset_name = "cais/mmlu"
try:
    dataset = load_dataset(dataset_name, "high_school_mathematics")
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Determine the correct split name based on inspection
split_name = 'validation'  # Adjust based on the actual split name in your dataset

# Select a subset of the dataset for testing (e.g., the first 10 examples from the chosen split)
subset = dataset[split_name].select(range(10))

# Function to generate the model's response
def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    # Increase max_length to 150 or adjust max_new_tokens as needed
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=150, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Evaluate the model on the selected subset
for example in subset:
    question = example['question']
    choices = example['choices']
    correct_answer = example['answer']

    # Prepare the input text for the model
    prompt = f"Question: {question}\nChoices: {choices}\nAnswer:"
    response = generate_response(prompt)
    print(f"Prompt: {prompt}")
    print(f"Model's Response: {response}")
    print(f"Correct Answer: {correct_answer}")
    print("=" * 80)
