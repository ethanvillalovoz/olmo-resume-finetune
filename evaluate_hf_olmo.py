# evaluate_hf_olmo.py
# Loads the fine-tuned HuggingFace-format OLMo model and tokenizer, runs inference on sample prompts, and prints the results.
# Usage: python evaluate_hf_olmo.py
# See README for details and sample output.

# Evaluate the fine-tuned OLMo model using HuggingFace Transformers
from transformers import OlmoForCausalLM, AutoTokenizer
import torch

def main():
    model_dir = "./my_hf_olmo_model"
    model = OlmoForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Example prompts to test generation
    prompts = [
        "Write a short professional summary for a software engineer:",
        "List three skills for a data scientist:",
        "Describe your experience with Python programming:",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,  # Deterministic output for debugging
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(f"\nPrompt: {prompt}")
        print("Response:")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
