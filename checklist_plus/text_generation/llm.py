import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from checklist_plus.config import cfg
from checklist_plus.text_generation.masked_lm import TextGenerator
from checklist_plus.text_generation.models import ParaphraseResponse, UniqueCompletions

logger = logging.getLogger(__name__)


class LLMTextGenerator(TextGenerator):
    """
    LLM-based TextGenerator that implements the same interface as TextGenerator
    but uses LLM for mask filling instead of masked language models.
    """

    def __init__(self,
                 llm_client: LLM | None = None,
                 openai_api_key: str | None = None,
                 model_name: str = "gpt-3.5-turbo",
                 **kwargs):
        """
        Initialize LLMTextGenerator.

        Parameters
        ----------
        llm_client : LLM, optional
            LangChain LLM client. If None, will try to use OpenAI.
        openai_api_key : str, optional
            OpenAI API key. If None, will try to get from environment.
        model_name : str
            Model name for OpenAI (default: "gpt-3.5-turbo")
        **kwargs
            Additional arguments
        """
        self.model_name = model_name

        # Use LangChain wrapper with _generate method for multiple completions
        self.llm_client = llm_client or self._setup_default_llm(openai_api_key, model_name)

        self.tokenizer = self._create_dummy_tokenizer()

    def _setup_default_llm(self, api_key: str | None, model_name: str) -> LLM:
        """Setup default LLM client using OpenAI."""
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                    "or pass openai_api_key parameter."
                )

        return ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            temperature=0.7,
            #max_tokens=100,
            )

    def _create_dummy_tokenizer(self):
        """Create a dummy tokenizer that mimics the interface of HuggingFace tokenizers."""
        class DummyTokenizer:
            def __init__(self):
                self.mask_token = "[MASK]"
                self.mask_token_id = 0
                self.unk_token = "[UNK]"

            def encode(self, text, add_special_tokens=True):
                # Simple word-based tokenization for compatibility
                return [0] + [hash(word) % 10000 for word in text.split()] + [1]

            def decode(self, token_ids):
                # Simple decoding for compatibility
                return " ".join([f"token_{id}" for id in token_ids if id not in [0, 1]])

            def tokenize(self, text):
                return text.split()

            def convert_tokens_to_ids(self, tokens):
                return [hash(token) % 10000 for token in tokens]

            def convert_ids_to_tokens(self, ids):
                return [f"token_{id}" for id in ids]

        return DummyTokenizer()

    def unmask_multiple(self, texts, n_completions=1, prompt_config=cfg.config.text_generation.llm.unmask_prompt, candidates=None, metric='avg', context=None, **kwargs):
        """
        Fill multiple mask tokens using LLM.

        Parameters
        ----------
        texts : List[str]
            List of texts with mask tokens
        n_completions : int
            Number of suggestions to generate per text
        candidates : List[str], optional
            Candidate words to consider (not used in LLM version)
        metric : str
            Metric for ranking (not used in LLM version)
        context : str, optional
            Topic or context to guide word generation (e.g., "science", "emotions", "technology")
            If None, generates diverse words covering various topics
        **kwargs
            Additional parameters

        Returns
        -------
        List[Tuple[List[str], str, float]]
            List of (words, full_text, score) tuples
        """
        all_results = []
        unique_completions = set()  # Track unique completions across all texts
        # print("texts:", texts)
        prompt_parts = []
        input_variables = ["n_completions", "text", "mask_count"]
        input_data = {
            "n_completions": n_completions,
        }
        prompt_parts.append(prompt_config.task_context)
        # print("candidates:", candidates)
        if candidates is not None:
            input_variables.append("candidates")
            input_data["candidates"] = ", ".join(candidates)
            prompt_parts.append(prompt_config.background_data.candidates)

        if context:
            input_variables.append("context")
            input_data["context"] = context
            prompt_parts.append(prompt_config.background_data.context)

        prompt_parts.extend([prompt_config.rules,
                        prompt_config.task,
                        prompt_config.thinking_step,
                        prompt_config.output_format])
        prompt_text = "\n".join(prompt_parts)
        # print("prompt_text:", context)
        completion_template = PromptTemplate(
            input_variables=input_variables,
            template=prompt_text
        )
        formatted_prompts = []
        for text in texts:
            # Count the number of masks in the text
            mask_count = text.count(self.tokenizer.mask_token)
            if mask_count == 0:
                continue
            input_data["mask_count"] = mask_count
            llm_text = text.replace(self.tokenizer.mask_token, "___")
            input_data["text"] = llm_text
            # Replace [MASK] with a more LLM-friendly placeholder
                                # Format the prompt with context
            formatted_prompt = completion_template.format(
                        **input_data
            )
            formatted_prompts.append(formatted_prompt)

        try:
            # Use structured output with Pydantic model
            structured_llm = self.llm_client.with_structured_output(UniqueCompletions)
            # print("here")
            responses = structured_llm.batch(formatted_prompts)
            # print("responses:", responses)

            # Extract completions from Pydantic model
            completions_all = [resp.completions for resp in responses if hasattr(resp, 'completions')]
            assert len(completions_all) == len(formatted_prompts), "Mismatch in number of responses"
            for completions, text in zip(completions_all, texts):
                # Process each completion set
                for i, completion_set in enumerate(completions[:n_completions]):
                    if len(completion_set) == mask_count:
                        # Replace masks one by one in order
                        full_text = text
                        for completion in completion_set:
                            # Clean the completion
                            cleaned = completion.strip(' .,!?;:')
                            full_text = full_text.replace(self.tokenizer.mask_token, cleaned, 1)
                        if full_text in unique_completions:
                            continue
                        unique_completions.add(full_text)
                        # Score decreases with position (first suggestion gets highest score)
                        score = 1.0 - (i * 0.01)  # Smaller decrement for more granular scoring
                        all_results.append((completion_set, full_text, score))
                    else:
                        logger.warning(f"Completion set {completion_set} has {len(completion_set)} items but expected {mask_count}")

        except Exception as e:
            logger.error(f"LLM unmask failed for text '{text}'", exc_info=True)
            # Fallback: return original text with empty completions
            all_results.append(([""] * mask_count, text, 0.0))

        # print('all_results:', all_results)
        # print(f'Total unique completions generated: {len(unique_completions)}')
        return all_results

    def unmask(self, text_with_mask, n_completions=10, candidates=None, context=None, **kwargs):
        """
        Fill mask tokens in a single text using LLM.

        Parameters
        ----------
        text_with_mask : str
            Text with mask tokens
        n_completions : int
            Number of suggestions to generate
        candidates : List[str], optional
            Candidate words to consider (not used in LLM version)
        context : str, optional
            Topic or context to guide word generation
        **kwargs
            Additional parameters

        Returns
        -------
        List[Tuple[List[str], str, float]]
            List of (words, full_text, score) tuples
        """
        # print("text_with_mask:", text_with_mask)
        # Use unmask_multiple with a single text and return the results
        results = self.unmask_multiple([text_with_mask], n_completions=n_completions,
                                     candidates=candidates, context=context, **kwargs)
        # print("Unmask results:", results[0])
        return results

    def paraphrase(self, texts, n_paraphrases=5, context=None, style=None,
                   length_preference=None, preserve_meaning=True,
                   temperature=0.7, **kwargs):
        """
        Generate paraphrases of text using LLM with structured output and batch processing.

        Parameters
        ----------
        texts : List[str] or str
            Text(s) to paraphrase
        n_paraphrases : int
            Number of paraphrases to generate per text
        context : str, optional
            Context or domain to guide paraphrasing (e.g., "formal", "casual", "academic", "business")
        style : str, optional
            Specific style instructions (e.g., "more formal", "simpler language", "technical")
        length_preference : str, optional
            Length preference: "shorter", "longer", or "similar" (default: similar)
        preserve_meaning : bool
            If True, emphasize preserving original meaning (default: True)
        temperature : float
            LLM temperature for creativity (0.0-1.0, default: 0.7)
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            List of paraphrased texts
        """
        if isinstance(texts, str):
            texts = [texts]

        # Determine length instruction based on preference
        length_instruction = ""
        if length_preference:
            if length_preference.lower() == "shorter":
                length_instruction = "Make the paraphrases more concise than the original"
            elif length_preference.lower() == "longer":
                length_instruction = "Make the paraphrases more detailed than the original"
            elif length_preference.lower() == "similar":
                length_instruction = "Keep the paraphrases similar in length to the original"

        prompt_parts = []
        input_variables = ["n_paraphrases", "text", "length_instruction"]
        input_data = {
            "n_paraphrases": n_paraphrases,
            "length_instruction": length_instruction
        }
        prompt_parts.extend([cfg.config.text_generation.llm.paraphrase_prompt.task_context])
        if style:
            prompt_parts.append(cfg.config.text_generation.llm.paraphrase_prompt.tone_context)
            input_variables.append("style")
            input_data["style"] = style
        if context:
            prompt_parts.append(cfg.config.text_generation.llm.paraphrase_prompt.background_data)
            input_variables.append("context")
            input_data["context"] = context
        prompt_parts.extend([cfg.config.text_generation.llm.paraphrase_prompt.rules,
                                cfg.config.text_generation.llm.paraphrase_prompt.task,
                                cfg.config.text_generation.llm.paraphrase_prompt.thinking_step,
                                cfg.config.text_generation.llm.paraphrase_prompt.output_format])

        prompt_text = "\n".join(prompt_parts)
        paraphrase_template = PromptTemplate(
            input_variables=input_variables,
            template=prompt_text
        )

        # Create all formatted prompts for batch processing
        formatted_prompts = []
        for text in texts:
            input_data["text"] = text
            formatted_prompt = paraphrase_template.format(**input_data)
            formatted_prompts.append(formatted_prompt)

        # Bind existing LLM client with specified temperature
        temp_llm = self.llm_client.bind(temperature=temperature)

        # Use structured output with Pydantic model
        structured_llm = temp_llm.with_structured_output(ParaphraseResponse)

        # Batch process all prompts
        all_paraphrases = []
        try:
            # Use batch method for efficient processing
            responses = structured_llm.batch(formatted_prompts)

            for i, response in enumerate(responses):
                original_text = texts[i]

                # Extract paraphrases from Pydantic model
                paraphrases = response.paraphrases if hasattr(response, 'paraphrases') else []

                # Filter out any paraphrases that are identical to the original
                filtered_paraphrases = [
                    para for para in paraphrases
                    if para.strip() and para.strip().lower() != original_text.strip().lower()
                ]

                # Ensure we have the requested number of paraphrases
                if len(filtered_paraphrases) < n_paraphrases:
                    # If we don't have enough unique paraphrases, pad with what we have
                    while len(filtered_paraphrases) < n_paraphrases and filtered_paraphrases:
                        filtered_paraphrases.append(filtered_paraphrases[0])

                all_paraphrases.extend(filtered_paraphrases[:n_paraphrases])

        except Exception as e:
            logger.error("LLM batch paraphrase failed", exc_info=True)
            raise

        return all_paraphrases

    def filter_options(self, texts, word, options, threshold=5, **kwargs):
        if type(texts) != list:
            texts = [texts]
        context = f"Only **{kwargs['type']}** for the word '{word}' in the given text."
        options = options + [word]
        in_all = set(options)
        orig_ret = []

        # Store original temperature and set to 0 for deterministic filtering
        original_temp = getattr(self.llm_client, 'temperature', None)
        if hasattr(self.llm_client, 'bind'):
            temp_llm = self.llm_client.bind(temperature=0)
            # Temporarily replace the client
            original_client = self.llm_client
            self.llm_client = temp_llm

        try:
            for text in texts:
                masked = re.sub(r'\b%s\b' % re.escape(word), self.tokenizer.mask_token, text)
                if masked == text:
                    continue
                ret = self.unmask(masked, beam_size=100, candidates=options, context=context, **kwargs)
                if not ret:
                    in_all = in_all.intersection(set())
                    continue
                non_word = [x for x in ret if np.all([y not in [self.tokenizer.unk_token, word] for y in x[0]])]
                score = [x for x in ret if np.all([y in [word, self.tokenizer.unk_token] for y in x[0]])]
                if score:
                    score = score[0][-1]
                # this will happen when the word is not in the vocabulary, in which case we don't look at the score
                else:
                    score = 0
                new_ret = [(x[0], x[1], score - x[2]) for x in non_word if score - x[2] < threshold]
                # print(text)
                # print(new_ret)
                # print()
                if text == texts[0]:
                    orig_ret = new_ret
                in_all = in_all.intersection({x[0][0] for x in new_ret})
        finally:
            # Restore original client
            if hasattr(self.llm_client, 'bind') and 'original_client' in locals():
                self.llm_client = original_client

        return [x for x in orig_ret if x[0][0] in in_all]
