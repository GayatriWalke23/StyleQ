"""
Example text samples for different writing styles.
"""
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class StyleSample:
    text: str
    description: str
    tags: List[str]

FORMAL_SAMPLES = [
    StyleSample(
        text="""
The study examines the correlation between socioeconomic status and academic achievement 
in urban environments. Through rigorous statistical analysis of longitudinal data, we 
demonstrate a significant positive relationship between household income and educational 
outcomes. Furthermore, the research indicates that early interventions can substantially 
mitigate the effects of economic disadvantage on academic performance.
""".strip(),
        description="Academic research paper excerpt",
        tags=["formal", "academic", "analytical"]
    ),
]

CASUAL_SAMPLES = [
    StyleSample(
        text="""
Hey! Just wanted to give you a quick update on how things are going. The party last 
weekend was amazing - you really missed out! Everyone was there and we had such a blast. 
BTW, are you free this Friday? We're thinking of grabbing dinner somewhere downtown. 
Let me know what you think! 😊
""".strip(),
        description="Casual message to friend",
        tags=["casual", "friendly", "informal"]
    ),
]

TECHNICAL_SAMPLES = [
    StyleSample(
        text="""
To install the package, run `pip install styleq` in your terminal. The library provides 
a high-level API for style transfer operations. Key features include:
- BERT-based style embeddings
- LoRA adapter for efficient fine-tuning
- Customizable generation parameters

For optimal performance, ensure you have CUDA-capable GPU with at least 8GB memory.
""".strip(),
        description="Technical documentation",
        tags=["technical", "instructional", "precise"]
    ),
]

CREATIVE_SAMPLES = [
    StyleSample(
        text="""
The old house stood silent against the twilight sky, its weathered windows reflecting 
the last golden rays of sunset. Ivy crept up its stone walls like ancient memories, 
each tendril a story waiting to be told. Inside, dust motes danced in shafts of fading 
light, painting ethereal patterns in the still air.
""".strip(),
        description="Creative writing excerpt",
        tags=["creative", "descriptive", "atmospheric"]
    ),
]

BUSINESS_SAMPLES = [
    StyleSample(
        text="""
I hope this email finds you well. Following our discussion last week, I am pleased to 
present our proposal for the Q3 marketing strategy. The attached document outlines our 
comprehensive approach, including budget allocations and projected ROI. Please review 
at your earliest convenience and let me know if you have any questions.
""".strip(),
        description="Business email",
        tags=["business", "professional", "formal"]
    ),
]

ALL_SAMPLES = {
    "formal": FORMAL_SAMPLES,
    "casual": CASUAL_SAMPLES,
    "technical": TECHNICAL_SAMPLES,
    "creative": CREATIVE_SAMPLES,
    "business": BUSINESS_SAMPLES,
}
