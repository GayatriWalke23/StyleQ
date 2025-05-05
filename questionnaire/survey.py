"""
StyleQ questionnaire implementation for analyzing writing style preferences.
Collects and processes user preferences for writing style customization.
"""
from typing import Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class StyleDimension(str, Enum):
    FORMALITY = "formality"
    COMPLEXITY = "complexity"
    TONE = "tone"
    STRUCTURE = "structure"
    ENGAGEMENT = "engagement"


class StyleOption(BaseModel):
    value: str
    description: str
    examples: List[str]


class StyleQuestion(BaseModel):
    id: str
    dimension: StyleDimension
    question: str
    description: str
    options: List[StyleOption]
    weight: float = 1.0


class StyleProfile(BaseModel):
    formality: float = Field(..., ge=0.0, le=1.0)
    complexity: float = Field(..., ge=0.0, le=1.0)
    tone: float = Field(..., ge=0.0, le=1.0)
    structure: float = Field(..., ge=0.0, le=1.0)
    engagement: float = Field(..., ge=0.0, le=1.0)


class StyleQuestionnaire:
    def __init__(self):
        self.questions = self._initialize_questions()
        self.responses: Dict[str, str] = {}

    def _initialize_questions(self) -> List[StyleQuestion]:
        """Initialize the style questionnaire with comprehensive questions."""
        return [
            StyleQuestion(
                id="formality_level",
                dimension=StyleDimension.FORMALITY,
                question="How formal should the writing style be?",
                description="This affects the level of professionalism and formality in the language.",
                options=[
                    StyleOption(
                        value="very_formal",
                        description="Highly professional and formal",
                        examples=["I would be grateful if you could review this proposal."]
                    ),
                    StyleOption(
                        value="somewhat_formal",
                        description="Professional but approachable",
                        examples=["Could you please take a look at this proposal?"]
                    ),
                    StyleOption(
                        value="neutral",
                        description="Balanced and neutral",
                        examples=["Please review this proposal."]
                    ),
                    StyleOption(
                        value="casual",
                        description="Friendly and informal",
                        examples=["Hey, mind checking out this proposal?"]
                    ),
                    StyleOption(
                        value="very_casual",
                        description="Very relaxed and conversational",
                        examples=["Hey! 👋 Would love your thoughts on this!"]
                    )
                ]
            ),
            StyleQuestion(
                id="complexity_level",
                dimension=StyleDimension.COMPLEXITY,
                question="What level of language complexity is preferred?",
                description="This determines the sophistication of vocabulary and sentence structure.",
                options=[
                    StyleOption(
                        value="simple",
                        description="Simple and clear language",
                        examples=["We need to finish this work by Friday."]
                    ),
                    StyleOption(
                        value="moderate",
                        description="Balanced vocabulary and structure",
                        examples=["We should aim to complete this project by the end of the week."]
                    ),
                    StyleOption(
                        value="complex",
                        description="Sophisticated language",
                        examples=["It would be optimal to conclude this endeavor prior to week's end."]
                    )
                ]
            ),
            StyleQuestion(
                id="tone_preference",
                dimension=StyleDimension.TONE,
                question="What tone should the writing convey?",
                description="This affects the emotional quality of the writing.",
                options=[
                    StyleOption(
                        value="professional",
                        description="Objective and business-focused",
                        examples=["The quarterly results show significant growth."]
                    ),
                    StyleOption(
                        value="friendly",
                        description="Warm and approachable",
                        examples=["Great news! Our numbers are looking really good this quarter!"]
                    ),
                    StyleOption(
                        value="enthusiastic",
                        description="Energetic and positive",
                        examples=["I'm thrilled to share our amazing quarterly results! 🎉"]
                    )
                ]
            )
        ]

    def get_questions(self) -> List[StyleQuestion]:
        """Return the list of style questions."""
        return self.questions

    def submit_response(self, question_id: str, response: str) -> bool:
        """Submit a response to a question."""
        for question in self.questions:
            if question.id == question_id:
                if any(opt.value == response for opt in question.options):
                    self.responses[question_id] = response
                    return True
        return False

    def _normalize_response(self, dimension: StyleDimension) -> float:
        """Convert categorical responses to normalized numerical values."""
        dimension_questions = [q for q in self.questions if q.dimension == dimension]
        if not dimension_questions:
            return 0.5  # Default to middle value if no questions for dimension
        
        total_weight = sum(q.weight for q in dimension_questions)
        weighted_sum = 0.0
        
        for question in dimension_questions:
            if question.id in self.responses:
                response = self.responses[question.id]
                # Convert categorical response to numerical value
                options = question.options
                response_idx = next(i for i, opt in enumerate(options) if opt.value == response)
                # For formality and complexity, "very_formal" (index 0) should map to 1.0
                if dimension in [StyleDimension.FORMALITY, StyleDimension.COMPLEXITY]:
                    normalized_value = 1.0 - (response_idx / (len(options) - 1))
                else:
                    normalized_value = response_idx / (len(options) - 1)
                weighted_sum += normalized_value * question.weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def get_style_profile(self) -> StyleProfile:
        """Generate a normalized style profile based on responses."""
        return StyleProfile(
            formality=self._normalize_response(StyleDimension.FORMALITY),
            complexity=self._normalize_response(StyleDimension.COMPLEXITY),
            tone=self._normalize_response(StyleDimension.TONE),
            structure=self._normalize_response(StyleDimension.STRUCTURE),
            engagement=self._normalize_response(StyleDimension.ENGAGEMENT)
        )
