from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List

class LLMGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.model.to(self.device)
    
    def generate_response(self, query: str, context: List[str], max_length: int = 512) -> str:
        """Generate response based on query and retrieved context"""
        # Combine context and query
        context_text = "\n".join(context)
        prompt = f"Context: {context_text}\n\nQuestion: {query}\n\nAnswer:"
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response

class SimpleLLMGenerator:
    """Alternative simple generator using Hugging Face pipeline"""
    def __init__(self):
        self.generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-small",
            tokenizer="microsoft/DialoGPT-small"
        )
    
    def generate_response(self, query: str, context: List[str], max_length: int = 200) -> str:
        """Generate simple response"""
        context_text = " ".join(context[:2])  # Use top 2 contexts
        prompt = f"Based on: {context_text[:300]}... Question: {query} Answer:"
        
        response = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7
        )[0]['generated_text']
        
        # Extract answer
        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        
        return "I couldn't generate a proper response based on the provided context."
