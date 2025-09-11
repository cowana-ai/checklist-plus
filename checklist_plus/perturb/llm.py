"""
LLM-enhanced perturbation methods for CheckList Plus.

This module provides LLM-powered text perturbation capabilities that extend
the rule-based perturbations in the base Perturb class.
"""

import logging
from typing import List, Optional, Union

from checklist_plus.text_generation import LLMTextGenerator

from .base import Perturb

logger = logging.getLogger(__name__)


class LLMPerturb(Perturb):
    """LLM-enhanced perturbation class with integrated text generation capabilities."""

    def __init__(self,
                 llm_text_generator: LLMTextGenerator | None = None,
                 openai_api_key: str | None = None,
                 model_name: str = "gpt-4o-mini",
                 fallback_to_rules: bool = True,
                 **kwargs):
        """
        Initialize LLMPerturb with integrated LLM capabilities.

        Parameters
        ----------
        llm_text_generator : LLMTextGenerator, optional
            Pre-configured LLM text generator. If None, will create a new one.
        openai_api_key : str, optional
            OpenAI API key for creating new LLM generator
        model_name : str, default "gpt-3.5-turbo"
            Model name for LLM generator
        fallback_to_rules : bool, default True
            Whether to fallback to rule-based methods when LLM fails
        **kwargs
            Additional arguments passed to LLMTextGenerator
        """
        super().__init__()

        if llm_text_generator is None:
            try:
                self.llm_generator = LLMTextGenerator(
                    openai_api_key=openai_api_key,
                    model_name=model_name,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM generator: {e}")
                if not fallback_to_rules:
                    raise
                self.llm_generator = None
        else:
            self.llm_generator = llm_text_generator

        self.fallback_to_rules = fallback_to_rules

    def _convert_to_string(self, text) -> str:
        """Convert spacy doc or other text format to string."""
        if hasattr(text, 'text'):
            return text.text
        return str(text)

    def _llm_with_fallback(self, texts: list[str], llm_method_name: str, rule_method_name: str, **kwargs):
        """
        Execute LLM method with optional fallback to rule-based method.

        Parameters
        ----------
        text : str or spacy.token.Doc
            Input text
        llm_method_name : str
            Name of the LLM method to call
        rule_method_name : str
            Name of the rule-based method to fallback to
        **kwargs
            Additional arguments for the methods

        Returns
        -------
        str or None
            Result from LLM or rule-based method
        """
        if self.llm_generator is not None:
            try:
                llm_method = getattr(self.llm_generator, llm_method_name)
                result = llm_method(texts, **kwargs)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"LLM method {llm_method_name} failed: {e}")

        # Fallback to rule-based method
        if self.fallback_to_rules:
            rule_method = getattr(super(), rule_method_name)
            return rule_method(texts)

        return None

    def add_negation_llm(self, texts: list[str], **kwargs) -> list[list[str]] | None:
        """
        Add negation using LLM with fallback to rule-based method.

        Parameters
        ----------
        text : str or spacy.token.Doc
            Input text to negate
        **kwargs
            Additional parameters for LLM generation

        Returns
        -------
        list[list[str]] or None
            Per-input list of negated variants, or None/empty lists on failure
        """
        return self._llm_with_fallback(
            texts,
            'negate_sentence_multiple',
            'add_negation',
            **kwargs
        )
