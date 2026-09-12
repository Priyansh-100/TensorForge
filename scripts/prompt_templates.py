#!/usr/bin/env python3
"""
Prompt template system for mini-GPT.

Supports:
- Few-shot prompting with examples
- Chain-of-thought prompting
- Structured output templates
- Conversation history management
- Prompt engineering utilities

Usage:
  python scripts/prompt_templates.py --template few_shot --task "sentiment" --input "I love this!"
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from transformer.gpt import CharTokenizer


class TemplateType(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STRUCTURED_OUTPUT = "structured_output"
    CONVERSATION = "conversation"
    REASONING = "reasoning"


@dataclass
class Example:
    """Single few-shot example."""
    input: str
    output: str
    reasoning: Optional[str] = None


@dataclass
class PromptTemplate:
    """Prompt template with placeholders."""
    name: str
    template_type: TemplateType
    system_prompt: str = ""
    instruction: str = ""
    examples: List[Example] = field(default_factory=list)
    input_placeholder: str = "{input}"
    output_placeholder: str = "{output}"
    reasoning_placeholder: str = "{reasoning}"
    prefix: str = ""
    suffix: str = ""
    separator: str = "\n\n"
    
    def format(self, input_text: str, **kwargs) -> str:
        """Format the prompt with input and optional variables."""
        parts = []
        
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")
        
        if self.instruction:
            parts.append(f"Instruction: {self.instruction}")
        
        # Add examples
        for ex in self.examples:
            if self.template_type == TemplateType.CHAIN_OF_THOUGHT and ex.reasoning:
                parts.append(
                    f"Input: {ex.input}\n"
                    f"Reasoning: {ex.reasoning}\n"
                    f"Output: {ex.output}"
                )
            else:
                parts.append(
                    f"Input: {ex.input}\n"
                    f"Output: {ex.output}"
                )
        
        # Add actual input
        parts.append(f"Input: {input_text}")
        
        if self.template_type == TemplateType.CHAIN_OF_THOUGHT:
            parts.append("Reasoning: Let me think step by step.")
            parts.append("Output:")
        else:
            parts.append("Output:")
        
        return self.separator.join(parts)


# Pre-built templates
TEMPLATES = {
    "sentiment_zero": PromptTemplate(
        name="sentiment_zero",
        template_type=TemplateType.ZERO_SHOT,
        instruction="Classify the sentiment of the following text as positive, negative, or neutral.",
        input_placeholder="{input}",
    ),
    
    "sentiment_few": PromptTemplate(
        name="sentiment_few",
        template_type=TemplateType.FEW_SHOT,
        instruction="Classify the sentiment of the following text as positive, negative, or neutral.",
        examples=[
            Example(input="I love this product!", output="positive"),
            Example(input="This is terrible quality.", output="negative"),
            Example(input="It's okay, nothing special.", output="neutral"),
        ],
    ),
    
    "sentiment_cot": PromptTemplate(
        name="sentiment_cot",
        template_type=TemplateType.CHAIN_OF_THOUGHT,
        instruction="Classify the sentiment of the following text as positive, negative, or neutral. Think step by step.",
        examples=[
            Example(
                input="I love this product!",
                reasoning="The word 'love' indicates strong positive emotion. Exclamation mark adds emphasis.",
                output="positive"
            ),
            Example(
                input="This is terrible quality.",
                reasoning="The word 'terrible' is strongly negative. 'quality' suggests product evaluation.",
                output="negative"
            ),
        ],
    ),
    
    "summarization": PromptTemplate(
        name="summarization",
        template_type=TemplateType.FEW_SHOT,
        instruction="Summarize the following text in one sentence.",
        examples=[
            Example(
                input="The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully when the fox leaped over it.",
                output="A fox jumps over a sleeping dog."
            ),
        ],
    ),
    
    "qa": PromptTemplate(
        name="qa",
        template_type=TemplateType.FEW_SHOT,
        system_prompt="You are a helpful assistant that answers questions accurately.",
        instruction="Answer the following question based on your knowledge.",
        examples=[
            Example(input="What is the capital of France?", output="Paris"),
            Example(input="Who wrote Romeo and Juliet?", output="William Shakespeare"),
        ],
    ),
    
    "code_generation": PromptTemplate(
        name="code_generation",
        template_type=TemplateType.FEW_SHOT,
        system_prompt="You are an expert programmer. Write clean, efficient code.",
        instruction="Write a Python function for the following task.",
        examples=[
            Example(
                input="Reverse a string",
                output="def reverse_string(s):\n    return s[::-1]"
            ),
        ],
    ),
    
    "reasoning_math": PromptTemplate(
        name="reasoning_math",
        template_type=TemplateType.CHAIN_OF_THOUGHT,
        instruction="Solve the following math problem step by step.",
        examples=[
            Example(
                input="If a train travels 60 miles in 1.5 hours, what is its average speed?",
                reasoning="Average speed = distance / time. Distance = 60 miles, time = 1.5 hours. 60 / 1.5 = 40.",
                output="40 mph"
            ),
        ],
    ),
    
    "conversation": PromptTemplate(
        name="conversation",
        template_type=TemplateType.CONVERSATION,
        system_prompt="You are a helpful, harmless, and honest AI assistant.",
        prefix="",
        suffix="",
        separator="\n",
    ),
    
    "shakespeare": PromptTemplate(
        name="shakespeare",
        template_type=TemplateType.ZERO_SHOT,
        instruction="Continue the following Shakespearean text in the same style:",
        suffix="",
    ),
    
    "translation": PromptTemplate(
        name="translation",
        template_type=TemplateType.FEW_SHOT,
        instruction="Translate the following English text to French.",
        examples=[
            Example(input="Hello, how are you?", output="Bonjour, comment allez-vous?"),
            Example(input="Thank you very much.", output="Merci beaucoup."),
        ],
    ),
    
    "structured_json": PromptTemplate(
        name="structured_json",
        template_type=TemplateType.STRUCTURED_OUTPUT,
        instruction="Extract information from the text and output as JSON with the specified schema.",
        examples=[
            Example(
                input="John Smith, age 30, lives in New York. He works as a software engineer.",
                output='{"name": "John Smith", "age": 30, "city": "New York", "occupation": "software engineer"}'
            ),
        ],
    ),
}


class PromptEngine:
    """High-level prompt engineering interface."""
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.conversation_history = []
    
    def generate(self, prompt: str, max_tokens: int = 100, 
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate completion for a prompt."""
        self.model.eval()
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            output = self.model.generate_cached(input_ids, max_tokens, temperature, top_p)
        
        generated = self.tokenizer.decode(output[0].tolist())
        return generated[len(prompt):]  # Return only new tokens
    
    def apply_template(self, template_name: str, input_text: str, **kwargs) -> str:
        """Apply a template to input text."""
        if template_name not in TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(TEMPLATES.keys())}")
        
        template = TEMPLATES[template_name]
        return template.format(input_text, **kwargs)
    
    def few_shot(self, task: str, input_text: str, examples: List[Example], 
                 instruction: str = "", max_tokens: int = 100, **gen_kwargs) -> str:
        """Run few-shot prompting with custom examples."""
        template = PromptTemplate(
            name=f"custom_{task}",
            template_type=TemplateType.FEW_SHOT,
            instruction=instruction or f"Complete the following {task} task.",
            examples=examples,
        )
        prompt = template.format(input_text)
        return self.generate(prompt, max_tokens, **gen_kwargs)
    
    def chain_of_thought(self, task: str, input_text: str, examples: List[Example],
                         instruction: str = "", max_tokens: int = 200, **gen_kwargs) -> str:
        """Run chain-of-thought prompting."""
        template = PromptTemplate(
            name=f"cot_{task}",
            template_type=TemplateType.CHAIN_OF_THOUGHT,
            instruction=instruction or f"Solve the following {task} step by step.",
            examples=examples,
        )
        prompt = template.format(input_text)
        return self.generate(prompt, max_tokens, **gen_kwargs)
    
    def converse(self, message: str, max_tokens: int = 100, **gen_kwargs) -> str:
        """Multi-turn conversation with history."""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Build conversation prompt
        prompt_parts = [TEMPLATES["conversation"].system_prompt]
        for turn in self.conversation_history:
            if turn["role"] == "user":
                prompt_parts.append(f"Human: {turn['content']}")
            else:
                prompt_parts.append(f"Assistant: {turn['content']}")
        prompt_parts.append("Assistant:")
        
        prompt = "\n".join(prompt_parts)
        response = self.generate(prompt, max_tokens, **gen_kwargs)
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


def demo():
    """Demo the prompt template system."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load model
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]
    
    from transformer.gpt import train
    model = train(
        CharTokenizer(text), train_data, val_data,
        epochs=1, block_size=128, d_model=64, num_heads=4, d_ff=256,
        num_layers=2, batch_size=32, save=False, rope=True,
        seed=0, num_pairs=2000, num_val_pairs=200
    )
    model.to(device)
    
    engine = PromptEngine(model, tokenizer, device)
    
    print("Prompt Template Demo")
    print("=" * 50)
    
    # Test different templates
    test_input = "To be, or not to be"
    
    print(f"\nInput: {test_input}")
    
    # Zero-shot
    prompt = engine.apply_template("shakespeare", test_input)
    print(f"\n[Shakespeare Template]\nPrompt:\n{prompt[:200]}...")
    result = engine.generate(prompt, max_tokens=50)
    print(f"Output: {result[:100]}...")
    
    # Few-shot with custom examples
    print("\n[Custom Few-Shot: Character Dialogue]")
    examples = [
        Example(input="Romeo: ", output="Romeo: O, she doth teach the torches to burn bright!"),
        Example(input="Hamlet: ", output="Hamlet: To be, or not to be, that is the question:"),
    ]
    result = engine.few_shot("dialogue", "Macbeth: ", examples, max_tokens=50)
    print(f"Output: Macbeth: {result[:100]}...")
    
    # Chain of thought
    print("\n[Chain of Thought: Simple Reasoning]")
    cot_examples = [
        Example(
            input="Complete: 'All the world's a ",
            reasoning="This is the famous opening of Jaques' monologue from As You Like It. The next word is 'stage'.",
            output="stage'"
        ),
    ]
    result = engine.chain_of_thought("completion", "Complete: 'To be, or not to ", cot_examples, max_tokens=50)
    print(f"Output: {result[:100]}...")
    
    print("\n" + "=" * 50)
    print("Available templates:", list(TEMPLATES.keys()))


if __name__ == "__main__":
    import torch
    demo()