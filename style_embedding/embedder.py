"""
Text style embedding functionality using transformer models.
Converts text and style profiles into meaningful vector representations.
"""
from typing import List, Dict, Optional, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from questionnaire.survey import StyleProfile


class StyleEmbedder:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the style embedder with a pre-trained transformer model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.device = device
        
        # Style projection layer
        self.style_projection = nn.Linear(5, embedding_dim)  # 5 style dimensions
        self.style_projection.to(device)

    def embed_text(self, text: str) -> np.ndarray:
        """Generate style embedding for input text."""
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling over sequence length
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings[0].cpu().numpy()

    def embed_style_profile(self, profile: StyleProfile) -> np.ndarray:
        """Convert style profile into embedding space."""
        # Convert profile to tensor
        style_vector = torch.tensor([
            profile.formality,
            profile.complexity,
            profile.tone,
            profile.structure,
            profile.engagement
        ], dtype=torch.float32).to(self.device)
        
        # Project style vector to embedding space
        with torch.no_grad():
            style_embedding = self.style_projection(style_vector)
            # Normalize embedding
            style_embedding = F.normalize(style_embedding, p=2, dim=0)
        
        return style_embedding.cpu().numpy()

    def compute_style_similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray]
    ) -> float:
        """Compute similarity between two text styles."""
        # Convert inputs to embeddings if they're strings
        if isinstance(text1, str):
            emb1 = self.embed_text(text1)
        else:
            emb1 = text1
            
        if isinstance(text2, str):
            emb2 = self.embed_text(text2)
        else:
            emb2 = text2
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

    def find_style_matches(
        self,
        target_style: Union[str, np.ndarray, StyleProfile],
        candidates: List[str],
        top_k: int = 3
    ) -> List[Dict[str, Union[str, float]]]:
        """Find the closest style matches among candidates."""
        # Convert target style to embedding
        if isinstance(target_style, str):
            target_emb = self.embed_text(target_style)
        elif isinstance(target_style, StyleProfile):
            target_emb = self.embed_style_profile(target_style)
        else:
            target_emb = target_style
        
        # Compute similarities for all candidates
        similarities = []
        for text in candidates:
            emb = self.embed_text(text)
            sim = self.compute_style_similarity(target_emb, emb)
            similarities.append((text, sim))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [
            {"text": text, "similarity": sim}
            for text, sim in similarities[:top_k]
        ]

    def interpolate_styles(
        self,
        style1: Union[str, np.ndarray, StyleProfile],
        style2: Union[str, np.ndarray, StyleProfile],
        alpha: float = 0.5
    ) -> np.ndarray:
        """Interpolate between two styles with mixing factor alpha."""
        # Convert styles to embeddings
        if isinstance(style1, str):
            emb1 = self.embed_text(style1)
        elif isinstance(style1, StyleProfile):
            emb1 = self.embed_style_profile(style1)
        else:
            emb1 = style1
            
        if isinstance(style2, str):
            emb2 = self.embed_text(style2)
        elif isinstance(style2, StyleProfile):
            emb2 = self.embed_style_profile(style2)
        else:
            emb2 = style2
        
        # Linear interpolation
        interpolated = (1 - alpha) * emb1 + alpha * emb2
        # Normalize
        interpolated = interpolated / np.linalg.norm(interpolated)
        return interpolated
