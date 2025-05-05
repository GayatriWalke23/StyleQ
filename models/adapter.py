"""
LoRA adapter implementation for style-conditioned text generation.
"""
from typing import Dict, List, Optional, Union, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
import numpy as np
from questionnaire.survey import StyleProfile
from style_embedding.embedder import StyleEmbedder


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA path: x -> A -> dropout -> B -> scale
        lora_output = (self.dropout(x) @ self.lora_A) @ self.lora_B
        return lora_output * self.scaling


class StyleAdapter(nn.Module):
    """Style-conditioned adapter using LoRA for text generation."""
    
    def __init__(
        self,
        base_model: str = "gpt2",
        style_dim: int = 768,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        
        # Set up padding token for GPT-2
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.to(device)
        
        # Initialize style embedder
        self.style_embedder = StyleEmbedder()
        
        # Get model dimensions
        self.hidden_size = self.model.config.n_embd
        
        # Style projection
        self.style_projection = nn.Linear(style_dim, self.hidden_size)
        
        # Create LoRA layers for each attention layer in GPT-2
        self.lora_layers = nn.ModuleDict()
        
        # Add LoRA layers to each transformer block
        for block_idx in range(len(self.model.transformer.h)):
            block = self.model.transformer.h[block_idx]
            
            # Add LoRA to query-key-value projection
            if hasattr(block.attn, "c_attn"):
                # For GPT-2's Conv1D layers, we need to use weight shape
                qkv_adapter = LoRALayer(
                    in_features=self.hidden_size,
                    out_features=3 * self.hidden_size,  # For q, k, v
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout
                )
                self.lora_layers[f"block_{block_idx}_qkv"] = qkv_adapter
            
            # Add LoRA to output projection
            if hasattr(block.attn, "c_proj"):
                out_adapter = LoRALayer(
                    in_features=self.hidden_size,
                    out_features=self.hidden_size,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout
                )
                self.lora_layers[f"block_{block_idx}_out"] = out_adapter
        
        self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        style_embedding: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Project style embedding if provided
        if style_embedding is not None:
            style_cond = self.style_projection(style_embedding)
        else:
            style_cond = None
            
        # Forward pass through base model with LoRA adaptations
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states
        
        # Apply LoRA adaptations to each layer
        for layer_idx, layer in enumerate(self.model.transformer.h):
            # Get hidden states for this layer
            layer_hidden = hidden_states[layer_idx]
            
            # Apply QKV LoRA
            qkv_lora = self.lora_layers[f"block_{layer_idx}_qkv"](layer_hidden)
            if style_cond is not None:
                # Add style conditioning
                qkv_lora = qkv_lora + style_cond.unsqueeze(1)
            
            # Apply output LoRA
            out_lora = self.lora_layers[f"block_{layer_idx}_out"](layer_hidden)
            if style_cond is not None:
                out_lora = out_lora + style_cond.unsqueeze(1)
            
            # Update hidden states
            hidden_states[layer_idx] = layer_hidden + qkv_lora + out_lora
        
        return {
            "logits": outputs.logits,
            "hidden_states": hidden_states,
            "loss": outputs.loss if labels is not None else None
        }

    def generate(
        self,
        prompt: str,
        style_embedding: torch.Tensor,
        max_length: int = 200,
        temperature: float = 0.7,
        preserve_content: bool = False
    ) -> str:
        """Generate text conditioned on style embedding."""
        # Encode prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate with style conditioning
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                encoder_hidden_states=style_embedding.unsqueeze(0)
            )
        
        # Decode and return generated text
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return generated_text
