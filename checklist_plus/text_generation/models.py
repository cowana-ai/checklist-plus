from typing import Optional, Union

from pydantic import BaseModel, Field, validator


class UniqueCompletions(BaseModel):
    """Pydantic model for generating unique text completions."""
    completions: list[list[str]] = Field(
        description="A list of completion sets. For single mask texts, each inner list has one completion. For multiple mask texts, each inner list contains completions in order for each mask position. Each completion should be either: 1) A single word, OR 2) A possessive noun phrase like 'goalkeeper's performance', 'student's homework', 'company's profits'. Avoid longer phrases like 'museum that I visited' or 'beautiful sunset today'. Example: [['game', 'it'], ['performance', 'food'], ['movie', 'service']] for two masks."
    )


class ParaphraseResponse(BaseModel):
    """Pydantic model for generating paraphrases of text."""
    paraphrases: list[str] = Field(
        description="A list of paraphrased versions of the input text. Each paraphrase should preserve the original meaning while using different words and sentence structures. Paraphrases should be natural, grammatically correct, and contextually appropriate."
    )


class NegationResponse(BaseModel):
    """Pydantic model for generating negated versions of text."""
    negated_sentences: list[str] = Field(
        description="A list of negation sets for batch processing. Each inner list contains negated versions of one input text. Each negated sentence should express the opposite meaning of the original while being grammatically correct and natural-sounding. Use appropriate negation words (not, never, no, don't, can't, etc.) and maintain the original style and formality level."
    )


class EntityDetectionResponse(BaseModel):
    """Response model for entity detection."""
    contains_entities: bool = Field(..., description="Whether the text contains entities of the specified type")
    entities: list[str] = Field(default_factory=list, description="List of detected entities, empty if none found")


class TextExample(BaseModel):
    """Single text generation example with input and structured output."""
    input: str = Field(..., description="Input text")
    output: ParaphraseResponse | NegationResponse | EntityDetectionResponse | UniqueCompletions | str = Field(
        ...,
        description="Expected output - can be a structured response model or simple string"
    )
    description: str | None = Field(None, description="Optional description of this example")

    @validator('input')
    def validate_input_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Input must be non-empty string")
        return v.strip()

    @validator('output')
    def validate_output(cls, v):
        if isinstance(v, str):
            if not v or not v.strip():
                raise ValueError("String output must be non-empty")
            return v.strip()
        # For Pydantic models, they handle their own validation
        return v


__all__ = ['TextExample', 'UniqueCompletions', 'ParaphraseResponse', 'NegationResponse', 'EntityDetectionResponse']
