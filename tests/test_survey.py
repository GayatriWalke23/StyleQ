"""
Unit tests for the StyleQ questionnaire module.
"""
import unittest
from questionnaire.survey import (
    StyleQuestionnaire,
    StyleDimension,
    StyleProfile
)


class TestStyleQuestionnaire(unittest.TestCase):
    def setUp(self):
        self.questionnaire = StyleQuestionnaire()

    def test_get_questions(self):
        """Test that questions are properly initialized."""
        questions = self.questionnaire.get_questions()
        self.assertTrue(len(questions) > 0)
        
        # Test first question structure
        first_question = questions[0]
        self.assertEqual(first_question.dimension, StyleDimension.FORMALITY)
        self.assertTrue(len(first_question.options) > 0)
        self.assertTrue(all(hasattr(opt, 'value') for opt in first_question.options))
        self.assertTrue(all(hasattr(opt, 'description') for opt in first_question.options))
        self.assertTrue(all(hasattr(opt, 'examples') for opt in first_question.options))

    def test_submit_valid_response(self):
        """Test submitting a valid response."""
        questions = self.questionnaire.get_questions()
        first_question = questions[0]
        valid_response = first_question.options[0].value
        
        success = self.questionnaire.submit_response(
            first_question.id,
            valid_response
        )
        self.assertTrue(success)

    def test_submit_invalid_response(self):
        """Test submitting an invalid response."""
        success = self.questionnaire.submit_response(
            "formality_level",
            "INVALID_OPTION"
        )
        self.assertFalse(success)

    def test_get_style_profile(self):
        """Test getting the style profile."""
        # Submit responses for all dimensions
        responses = {
            "formality_level": "very_formal",
            "complexity_level": "complex",
            "tone_preference": "professional"
        }
        
        for question_id, response in responses.items():
            success = self.questionnaire.submit_response(question_id, response)
            self.assertTrue(success)
        
        profile = self.questionnaire.get_style_profile()
        self.assertIsInstance(profile, StyleProfile)
        
        # Test that all dimensions are present
        self.assertTrue(hasattr(profile, 'formality'))
        self.assertTrue(hasattr(profile, 'complexity'))
        self.assertTrue(hasattr(profile, 'tone'))
        self.assertTrue(hasattr(profile, 'structure'))
        self.assertTrue(hasattr(profile, 'engagement'))
        
        # Test that values are normalized between 0 and 1
        self.assertTrue(0 <= profile.formality <= 1)
        self.assertTrue(0 <= profile.complexity <= 1)
        self.assertTrue(0 <= profile.tone <= 1)
        self.assertTrue(0 <= profile.structure <= 1)
        self.assertTrue(0 <= profile.engagement <= 1)

    def test_normalize_response(self):
        """Test response normalization."""
        # Submit a response that should result in maximum formality
        self.questionnaire.submit_response("formality_level", "very_formal")
        formality = self.questionnaire._normalize_response(StyleDimension.FORMALITY)
        self.assertEqual(formality, 1.0)
        
        # Submit a response that should result in minimum formality
        self.questionnaire.submit_response("formality_level", "very_casual")
        formality = self.questionnaire._normalize_response(StyleDimension.FORMALITY)
        self.assertEqual(formality, 0.0)


if __name__ == "__main__":
    unittest.main()
