import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Check for GPU availability
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No GPU available")

# Define the path to your saved model
checkpoint_path = "./results/checkpoint-256"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

# Set up the conversational pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to interact with the chatbot
def interact_with_chatbot(prompt, max_length=200):
    response = chatbot(prompt, max_length=max_length, do_sample=True, top_p=0.9, temperature=0.9)
    return response[0]['generated_text']

# Continuous interaction loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = interact_with_chatbot(user_input)
    print("Chatbot:", response)


