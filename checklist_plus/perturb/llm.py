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

    def add_negation_llm(self, texts: list[str], **kwargs) -> str | None:
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
        str or None
            Negated text, or None if negation not possible
        """
        return self._llm_with_fallback(
            texts,
            'negate_sentence_multiple',
            'add_negation',
            **kwargs
        )


class SmartPerturb:
    """
    Convenience wrapper that automatically chooses between rule-based and LLM methods.
    """

    def __init__(self, prefer_llm: bool = True, **llm_kwargs):
        """
        Initialize SmartPerturb with automatic method selection.

        Parameters
        ----------
        prefer_llm : bool, default True
            Whether to prefer LLM methods over rule-based when available
        **llm_kwargs
            Arguments passed to LLMPerturb if LLM is preferred
        """
        self.prefer_llm = prefer_llm

        if prefer_llm:
            try:
                self.perturber = LLMPerturb(**llm_kwargs)
                self.has_llm = self.perturber.llm_generator is not None
            except Exception:
                self.perturber = Perturb()
                self.has_llm = False
        else:
            self.perturber = Perturb()
            self.has_llm = False

    def add_negation(self, text, **kwargs):
        """Smart negation addition - uses LLM if available, otherwise rule-based."""
        if self.has_llm:
            return self.perturber.add_negation_llm(text, **kwargs)
        return self.perturber.add_negation(text)

    def remove_negation(self, text, **kwargs):
        """Smart negation removal - uses LLM if available, otherwise rule-based."""
        if self.has_llm:
            return self.perturber.remove_negation_llm(text, **kwargs)
        return self.perturber.remove_negation(text)

    def paraphrase(self, text, **kwargs):
        """Generate paraphrases - uses LLM if available."""
        if self.has_llm:
            return self.perturber.paraphrase_llm(text, **kwargs)
        logger.warning("Paraphrasing requires LLM capabilities")
        return []

    def change_style(self, text, style, **kwargs):
        """Change text style - uses LLM if available."""
        if self.has_llm:
            return self.perturber.rephrase_with_style_llm(text, style, **kwargs)
        logger.warning("Style changing requires LLM capabilities")
        return None


# Convenience functions for backward compatibility
def create_llm_perturber(openai_api_key: str | None = None,
                        model_name: str = "gpt-3.5-turbo",
                        **kwargs) -> LLMPerturb:
    """
    Convenience function to create an LLMPerturb instance.

    Parameters
    ----------
    openai_api_key : str, optional
        OpenAI API key
    model_name : str, default "gpt-3.5-turbo"
        Model name for LLM
    **kwargs
        Additional arguments for LLMPerturb

    Returns
    -------
    LLMPerturb
        Configured LLMPerturb instance
    """
    return LLMPerturb(
        openai_api_key=openai_api_key,
        model_name=model_name,
        **kwargs
    )


def create_smart_perturber(prefer_llm: bool = True, **kwargs) -> SmartPerturb:
    """
    Convenience function to create a SmartPerturb instance.

    Parameters
    ----------
    prefer_llm : bool, default True
        Whether to prefer LLM methods
    **kwargs
        Additional arguments for underlying perturber

    Returns
    -------
    SmartPerturb
        Configured SmartPerturb instance
    """
    return SmartPerturb(prefer_llm=prefer_llm, **kwargs)
