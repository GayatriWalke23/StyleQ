"""
Unit tests for the StyleQ embedder module.
"""
import unittest
import numpy as np
from style_embedding.embedder import StyleEmbedder
from questionnaire.survey import StyleProfile


class TestStyleEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = StyleEmbedder()
        cls.test_texts = [
            "This is a formal business document.",
            "hey what's up! how r u doing? 😊",
            "The results demonstrate a significant correlation between variables."
        ]
        cls.test_profile = StyleProfile(
            formality=0.8,
            complexity=0.7,
            tone=0.6,
            structure=0.5,
            engagement=0.4
        )

    def test_embed_text(self):
        """Test text embedding generation."""
        embedding = self.embedder.embed_text(self.test_texts[0])
        
        # Check embedding shape and type
        self.assertEqual(embedding.shape, (self.embedder.embedding_dim,))
        self.assertIsInstance(embedding, np.ndarray)
        
        # Check embedding is normalized
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=6)

    def test_embed_style_profile(self):
        """Test style profile embedding."""
        embedding = self.embedder.embed_style_profile(self.test_profile)
        
        # Check embedding shape and type
        self.assertEqual(embedding.shape, (self.embedder.embedding_dim,))
        self.assertIsInstance(embedding, np.ndarray)
        
        # Check embedding is normalized
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=6)

    def test_compute_style_similarity(self):
        """Test style similarity computation."""
        # Same text should have perfect similarity
        sim = self.embedder.compute_style_similarity(
            self.test_texts[0],
            self.test_texts[0]
        )
        self.assertAlmostEqual(sim, 1.0, places=6)
        
        # Different texts should have lower similarity
        sim = self.embedder.compute_style_similarity(
            self.test_texts[0],
            self.test_texts[1]
        )
        self.assertLess(sim, 1.0)
        self.assertGreaterEqual(sim, -1.0)

    def test_find_style_matches(self):
        """Test finding style matches."""
        matches = self.embedder.find_style_matches(
            self.test_texts[0],
            self.test_texts,
            top_k=2
        )
        
        # Check structure and length
        self.assertEqual(len(matches), 2)
        self.assertIn("text", matches[0])
        self.assertIn("similarity", matches[0])
        
        # First match should be the query itself
        self.assertEqual(matches[0]["text"], self.test_texts[0])
        self.assertAlmostEqual(matches[0]["similarity"], 1.0, places=6)

    def test_interpolate_styles(self):
        """Test style interpolation."""
        # Test with text inputs
        interpolated = self.embedder.interpolate_styles(
            self.test_texts[0],
            self.test_texts[1],
            alpha=0.5
        )
        
        # Check shape and normalization
        self.assertEqual(interpolated.shape, (self.embedder.embedding_dim,))
        self.assertAlmostEqual(np.linalg.norm(interpolated), 1.0, places=6)
        
        # Test with style profile
        interpolated = self.embedder.interpolate_styles(
            self.test_profile,
            self.test_texts[0],
            alpha=0.3
        )
        
        # Check shape and normalization
        self.assertEqual(interpolated.shape, (self.embedder.embedding_dim,))
        self.assertAlmostEqual(np.linalg.norm(interpolated), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
