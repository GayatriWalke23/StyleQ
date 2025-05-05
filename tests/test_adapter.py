"""
Unit tests for the StyleQ LoRA adapter module.
"""
import unittest
import torch
import numpy as np
from models.adapter import LoRALayer, StyleAdapter
from questionnaire.survey import StyleProfile


class TestLoRALayer(unittest.TestCase):
    def setUp(self):
        self.in_features = 768
        self.out_features = 768
        self.rank = 8
        self.batch_size = 2
        self.seq_length = 10
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank
        )

    def test_initialization(self):
        """Test LoRA layer initialization."""
        self.assertEqual(self.lora.lora_A.shape, (self.in_features, self.rank))
        self.assertEqual(self.lora.lora_B.shape, (self.rank, self.out_features))

    def test_forward(self):
        """Test LoRA layer forward pass."""
        x = torch.randn(self.batch_size, self.seq_length, self.in_features)
        output = self.lora(x)
        
        # Check output shape
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_length, self.out_features)
        )
        
        # Check scaling is applied
        self.assertTrue(torch.all(output == output * self.lora.scaling))


class TestStyleAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = StyleAdapter(base_model="gpt2")
        cls.test_text = "This is a test prompt."
        cls.test_profile = StyleProfile(
            formality=0.8,
            complexity=0.7,
            tone=0.6,
            structure=0.5,
            engagement=0.4
        )

    def test_initialization(self):
        """Test adapter initialization."""
        self.assertIsNotNone(self.adapter.model)
        self.assertIsNotNone(self.adapter.tokenizer)
        self.assertIsNotNone(self.adapter.style_embedder)
        self.assertGreater(len(self.adapter.lora_layers), 0)

    def test_style_injection(self):
        """Test style injection."""
        batch_size = 2
        seq_length = 10
        hidden_dim = self.adapter.hidden_size
        style_dim = 768
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_dim)
        style_embedding = torch.randn(batch_size, style_dim)
        
        output = self.adapter._inject_style(hidden_states, style_embedding)
        
        # Check output shape
        self.assertEqual(
            output.shape,
            (batch_size, seq_length, hidden_dim)
        )

    def test_forward(self):
        """Test forward pass."""
        # Prepare inputs
        inputs = self.adapter.tokenizer(
            self.test_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Forward pass with text style
        outputs = self.adapter(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            style=self.test_text
        )
        
        # Check outputs
        self.assertIn("logits", outputs)
        self.assertIn("hidden_states", outputs)
        self.assertEqual(
            outputs["logits"].shape[0],
            inputs["input_ids"].shape[0]
        )

    def test_generate(self):
        """Test text generation."""
        # Generate with text style
        outputs_text = self.adapter.generate(
            prompt=self.test_text,
            style=self.test_text,
            max_length=20
        )
        self.assertEqual(len(outputs_text), 1)
        self.assertIsInstance(outputs_text[0], str)
        
        # Generate with style profile
        outputs_profile = self.adapter.generate(
            prompt=self.test_text,
            style=self.test_profile,
            max_length=20
        )
        self.assertEqual(len(outputs_profile), 1)
        self.assertIsInstance(outputs_profile[0], str)

    def test_save_load_adapter(self, tmp_path="test_adapter.pt"):
        """Test saving and loading adapter weights."""
        # Save adapter
        self.adapter.save_adapter(tmp_path)
        
        # Create new adapter and load weights
        new_adapter = StyleAdapter(base_model="gpt2")
        new_adapter.load_adapter(tmp_path)
        
        # Check that weights are loaded correctly
        for (name1, param1), (name2, param2) in zip(
            self.adapter.lora_layers.named_parameters(),
            new_adapter.lora_layers.named_parameters()
        ):
            self.assertTrue(torch.allclose(param1, param2))


if __name__ == "__main__":
    unittest.main()
