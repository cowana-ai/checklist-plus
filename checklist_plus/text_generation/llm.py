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
from checklist_plus.text_generation.models import (
    EntityDetectionResponse,
    NegationResponse,
    ParaphraseResponse,
    TextExample,
    UniqueCompletions,
)

logger = logging.getLogger(__name__)


class LLMTextGenerator(TextGenerator):
    """
    LLM-based TextGenerator that implements the same interface as TextGenerator
    but uses LLM for mask filling instead of masked language models.
    """

    def __init__(self,
                 llm_client: LLM | None = None,
                 openai_api_key: str | None = None,
                 model_name: str = "gpt-4o-mini",
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
            Model name for OpenAI (default: "gpt-4o-mini")
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

    @property
    def entity_detection_examples(self):
        """Built-in examples for entity detection using structured TextExample format."""
        from checklist_plus.text_generation.models import (
            EntityDetectionResponse,
            TextExample,
        )

        return [
            # Example 1: Brand names detection
            TextExample(
                input="text: I love my iPhone and Tesla car, entity_type=BRAND",
                output=EntityDetectionResponse(
                    contains_entities=True,
                    entities=["iPhone", "Tesla"]
                ),
                description="BRAND"
            ),

            # Example 2: Person names detection
            TextExample(
                input="text: John called Mary yesterday to discuss the project, entity_type=PERSON",
                output=EntityDetectionResponse(
                    contains_entities=True,
                    entities=["John", "Mary"]
                ),
                description="PERSON"
            ),

            # Example 3: No entities found
            TextExample(
                input="text: The weather is nice today and I feel great, entity_type=PERSON",
                output=EntityDetectionResponse(
                    contains_entities=False,
                    entities=[]
                ),
                description="PERSON"
            )
        ]

    def _process_examples(self, examples):
        """Process various example formats into formatted string for prompt."""
        if not examples:
            return ""

        from checklist_plus.text_generation.models import (
            EntityDetectionResponse,
            NegationResponse,
            ParaphraseResponse,
            TextExample,
            UniqueCompletions,
        )

        formatted_examples = []

        if isinstance(examples, list):
            for example in examples:
                if isinstance(example, TextExample):
                    # Handle structured output
                    if isinstance(example.output, str):
                        formatted_examples.append(f"{example.input} -> {example.output}")
                    elif isinstance(example.output, ParaphraseResponse):
                        paraphrases_str = ", ".join(example.output.paraphrases)
                        formatted_examples.append(f"{example.input} -> Paraphrases: [{paraphrases_str}]")
                    elif isinstance(example.output, NegationResponse):
                        negations_str = ", ".join(example.output.negated_sentences)
                        formatted_examples.append(f"{example.input} -> Negations: [{negations_str}]")
                    elif isinstance(example.output, EntityDetectionResponse):
                        entities_str = ", ".join(example.output.entities) if example.output.entities else "none"
                        contains_entities = "Yes" if example.output.contains_entities else "No"
                        formatted_examples.append(f"{example.input} -> Contains Entities: {contains_entities}, Entities: {entities_str}")
                    elif isinstance(example.output, UniqueCompletions):
                        completions_str = str(example.output.completions)
                        formatted_examples.append(f"{example.input} -> Completions: {completions_str}")
                    else:
                        formatted_examples.append(f"{example.input} -> {str(example.output)}")
                elif isinstance(example, tuple) and len(example) == 2:
                    formatted_examples.append(f"{example[0]} -> {example[1]}")
                elif isinstance(example, dict):
                    inp = example.get('input') or example.get('original', '')
                    out = example.get('output') or example.get('paraphrase', '')
                    if inp and out:
                        formatted_examples.append(f"{inp} -> {out}")

        if formatted_examples:
            return "Here are some examples of how to respond:\n\n" + "\n\n".join(formatted_examples) + "\n\n"
        return ""

    def _build_prompt_parts(self, prompt_config, examples=None, candidates=None, context=None,
                           style=None, **kwargs):
        """Build prompt parts in a standardized way."""
        prompt_parts = []
        input_variables = []
        input_data = {}

        # Always start with task context
        prompt_parts.append(prompt_config.task_context)

        # Add candidates if provided
        if candidates is not None:
            input_variables.append("candidates")
            input_data["candidates"] = ", ".join(candidates)
            if hasattr(prompt_config, 'background_data') and hasattr(prompt_config.background_data, 'candidates'):
                prompt_parts.append(prompt_config.background_data.candidates)

        # Add context if provided
        if context:
            input_variables.append("context")
            input_data["context"] = context
            if hasattr(prompt_config, 'background_data') and hasattr(prompt_config.background_data, 'context'):
                prompt_parts.append(prompt_config.background_data.context)

        # Add style if provided (for paraphrase)
        if style:
            input_variables.append("style")
            input_data["style"] = style
            if hasattr(prompt_config, 'tone_context'):
                prompt_parts.append(prompt_config.tone_context)

        # Add background data sections that should always be included
        if hasattr(prompt_config, 'background_data'):
            # For negation, add preserve_style
            if hasattr(prompt_config.background_data, 'preserve_style'):
                prompt_parts.append(prompt_config.background_data.preserve_style)
            # For paraphrase with context, add background_data
            elif context and not hasattr(prompt_config.background_data, 'context'):
                prompt_parts.append(prompt_config.background_data)

        # Add rules
        if hasattr(prompt_config, 'rules'):
            prompt_parts.append(prompt_config.rules)

        # Add examples if provided
        examples_text = self._process_examples(examples)
        if examples_text:
            prompt_parts.append(examples_text)

        # Add core components
        if hasattr(prompt_config, 'task'):
            prompt_parts.append(prompt_config.task)
        if hasattr(prompt_config, 'thinking_step'):
            prompt_parts.append(prompt_config.thinking_step)
        if hasattr(prompt_config, 'output_format'):
            prompt_parts.append(prompt_config.output_format)

        return prompt_parts, input_variables, input_data

    def _build_prompt_text(self, prompt_config, examples=None, candidates=None, context=None,
                          style=None, **additional_vars):
        """Build complete prompt text and return template components."""
        prompt_parts, input_variables, input_data = self._build_prompt_parts(
            prompt_config, examples=examples, candidates=candidates,
            context=context, style=style, **additional_vars
        )

        # Add any additional variables
        for key, value in additional_vars.items():
            if key not in input_data:
                input_variables.append(key)
                input_data[key] = value

        prompt_text = "\n".join(prompt_parts)

        return prompt_text, input_variables, input_data

    def unmask_multiple(self, texts, n_completions=1, prompt_config=cfg.config.text_generation.llm.unmask_prompt,
                        candidates: list[str]=None, metric='avg', context: str=None,
                        examples: list[TextExample | tuple[str, str] | dict[str, str]] = None, **kwargs):
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

        # Build prompt using helper function
        prompt_text, input_variables, input_data = self._build_prompt_text(
            prompt_config,
            examples=examples,
            candidates=candidates,
            context=context,
            n_completions=n_completions,
            text="",  # Will be filled per text
            mask_count=0  # Will be filled per text
        )

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

    def unmask(self, text_with_mask, n_completions=10, candidates: list[str]=None, context: str=None, examples: list[TextExample | tuple[str, str] | dict[str, str]] = None, **kwargs):
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
                                     candidates=candidates, context=context, examples=examples, **kwargs)
        # print("Unmask results:", results[0])
        return results

    def paraphrase(self, texts, n_paraphrases=5, examples: list[TextExample | tuple[str, str] | dict[str, str]] = None,
                   context=None, style=None,
                   prompt_config=cfg.config.text_generation.llm.paraphrase_prompt,
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
        examples : List[TextExample], List[Tuple], List[Dict], str, optional
            Examples to guide paraphrasing style. Can be:
            - List of TextExample objects
            - List of (input, output) tuples
            - List of dictionaries with input/output keys
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

        # Build prompt using helper function
        prompt_text, input_variables, input_data = self._build_prompt_text(
            prompt_config,
            examples=examples,
            context=context,
            style=style,
            n_paraphrases=n_paraphrases,
            text="",  # Will be filled per text
            length_instruction=length_instruction
        )

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

    def negate_sentence_multiple(self, texts, n_variations=1,
                                 prompt_config=cfg.config.text_generation.llm.negation_prompt,
                                 context: str=None,
                                 examples: list[TextExample | tuple[str, str] | dict[str, str]] = None, **kwargs):
        """
        Generate negated versions of multiple sentences using LLM with batch processing.

        Parameters
        ----------
        texts : List[str]
            List of texts to negate
        n_variations : int
            Number of negated variations to generate per text
        prompt_config : object, optional
            Configuration for negation prompts. If None, uses default from config.
        context : str, optional
            Context to guide negation (e.g., "formal", "casual", "academic")
        **kwargs
            Additional parameters

        Returns
        -------
        List[List[str]]
            List of lists, where each inner list contains negated versions of the corresponding input text
        """

        # Build prompt using helper function
        prompt_text, input_variables, input_data = self._build_prompt_text(
            prompt_config,
            examples=examples,
            context=context,
            n_variations=n_variations,
            text=""  # Will be filled per text
        )

        negation_template = PromptTemplate(
            input_variables=input_variables,
            template=prompt_text
        )

        # Create all formatted prompts for batch processing
        formatted_prompts = []
        for text in texts:
            input_data["text"] = text
            formatted_prompt = negation_template.format(**input_data)
            formatted_prompts.append(formatted_prompt)

        # Use structured output with Pydantic model
        structured_llm = self.llm_client.with_structured_output(NegationResponse)

        # Batch process all prompts
        all_negations = []
        try:
            # Use batch method for efficient processing
            responses = structured_llm.batch(formatted_prompts)
            # logger.warning(f"response: {responses}")
            for i, response in enumerate(responses):
                original_text = texts[i]

                # Extract negated sentences from Pydantic model
                negated_sentences = response.negated_sentences if hasattr(response, 'negated_sentences') else []

                # Filter out any sentences that are identical to the original
                filtered_negations = [
                    neg for neg in negated_sentences
                    if neg.strip() and neg.strip().lower() != original_text.strip().lower()
                ]

                # Ensure we have the requested number of negations
                if len(filtered_negations) < n_variations:
                    logger.warning(f"Not enough unique negations for text: {original_text}")
                # If we still don't have any, return empty list for this text
                if not filtered_negations:
                    filtered_negations = []

                all_negations.append(filtered_negations[:n_variations])

        except Exception as e:
            logger.error(f"LLM batch negation failed {e}", exc_info=True)
            # Return empty lists for all texts on failure
            all_negations = [[] for _ in texts]

        return all_negations

    def negate_sentence(self, text, n_variations=1, context=None, **kwargs):
        """
        Generate negated versions of a sentence using LLM.

        Parameters
        ----------
        text : str or List[str]
            Input text(s) to negate. If string, will process as single text.
            If list, will process all texts in batch.
        n_variations : int
            Number of negated variations to generate per text
        context : str, optional
            Context to guide negation (e.g., "formal", "casual", "academic")
        **kwargs
            Additional parameters

        Returns
        -------
        List[str] or List[List[str]]
            If input was string: List of negated sentences
            If input was list: List of lists, where each inner list contains negated versions
        """
        # Handle both single text and batch processing
        if isinstance(text, str):
            texts = [text]
            return_single = True
        else:
            texts = text
            return_single = False

        # Use the batch method
        results = self.negate_sentence_multiple(texts, n_variations=n_variations, context=context, **kwargs)

        if return_single:
            return results[0] if results else []
        else:
            return results

    def detect_entities(self, texts, entity_type,
                        prompt_config=cfg.config.text_generation.llm.entity_detection_prompt,
                       examples: list[TextExample | tuple[str, str] | dict[str, str]] = None,
                       context=None, temperature=0.0, **kwargs):
        """
        Detect if texts contain entities of a specific type and return the detected entities.

        Parameters
        ----------
        texts : str or List[str]
            Text(s) to analyze
        entity_type : str
            Type of entity to detect (e.g., "brand names", "person names", "animals")
        examples : List[TextExample], List[Tuple], List[Dict], optional
            Examples showing what entities of this type look like
        context : str, optional
            Additional context to guide entity detection
        temperature : float
            LLM temperature for detection (default: 0.3)
        **kwargs
            Additional parameters

        Returns
        -------
        List[Dict] or Dict
            Detection results with format:
            {
                "text": str,                    # Original text
                "contains_entities": bool,      # True if entities found
                "entities": List[str]           # List of detected entities
            }

        Examples
        --------
        >>> result = generator.detect_entities("I love my iPhone and Tesla", "brand names")
        >>> print(result)
        {
            "text": "I love my iPhone and Tesla",
            "contains_entities": True,
            "entities": ["iPhone", "Tesla"]
        }
        """
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False

        # Add default examples if requested
        if examples is None:
            examples = self.entity_detection_examples

        # Build prompt using helper function
        prompt_text, input_variables, input_data = self._build_prompt_text(
            prompt_config,
            examples=examples,
            context=context,
            entity_type=entity_type,
            text=""  # Will be filled per text
        )

        detection_template = PromptTemplate(
            input_variables=input_variables,
            template=prompt_text
        )

        # Create formatted prompts
        formatted_prompts = []
        for text in texts:
            input_data["text"] = text
            formatted_prompt = detection_template.format(**input_data)
            formatted_prompts.append(formatted_prompt)

        # Use lower temperature for consistent detection
        temp_llm = self.llm_client.bind(temperature=temperature)
        structured_llm = temp_llm.with_structured_output(EntityDetectionResponse)

        results = []
        try:
            responses = structured_llm.batch(formatted_prompts)

            for i, response in enumerate(responses):
                original_text = texts[i]

                result = {
                    "text": original_text,
                    "contains_entities": response.contains_entities if hasattr(response, 'contains_entities') else False,
                    "entities": response.entities if hasattr(response, 'entities') else []
                }

                results.append(result)

        except Exception as e:
            logger.error(f"LLM entity detection failed: {e}", exc_info=True)
            # Fallback results
            fallback_result = {
                "text": text,
                "contains_entities": False,
                "entities": []
            }
            results = [fallback_result for text in texts]

        if return_single:
            return results[0] if results else fallback_result
        else:
            return results

    def mask_detected_entities(self, text, entities, mask_token="{mask}",
                              case_sensitive=False, whole_words_only=True):
        """
        Manually mask detected entities in text.

        Parameters
        ----------
        text : str
            Original text
        entities : List[str]
            List of entities to mask
        mask_token : str
            Token to use for masking (default: "[MASK]")
        case_sensitive : bool
            Whether to perform case-sensitive matching (default: False)
        whole_words_only : bool
            Whether to match whole words only (default: True)

        Returns
        -------
        str
            Text with entities masked

        Examples
        --------
        >>> masked = generator.mask_detected_entities(
        ...     "I love my iPhone and Tesla car",
        ...     ["iPhone", "Tesla"]
        ... )
        >>> print(masked)
        "I love my [MASK] and [MASK] car"
        """
        masked_text = text

        for entity in entities:
            if not entity.strip():
                continue

            if whole_words_only:
                pattern = r'\b' + re.escape(entity) + r'\b'
            else:
                pattern = re.escape(entity)

            flags = 0 if case_sensitive else re.IGNORECASE
            masked_text = re.sub(pattern, mask_token, masked_text, flags=flags)

        return masked_text

    def detect_and_mask_entities(self, texts, entity_type,
                                examples: list[TextExample | tuple[str, str] | dict[str, str]] = None,
                                context=None, temperature=0.0, mask_token="{mask}",
                                case_sensitive=False, whole_words_only=True, **kwargs):
        """
        Detect entities and mask them in one convenient function.

        Parameters
        ----------
        texts : str or List[str]
            Text(s) to analyze and mask
        entity_type : str
            Type of entity to detect (e.g., "brand names", "person names", "animals")
        examples : List[TextExample], List[Tuple], List[Dict], optional
            Examples showing what entities of this type look like
        context : str, optional
            Additional context to guide entity detection
        temperature : float
            LLM temperature for detection (default: 0.3)
        mask_token : str
            Token to use for masking (default: "[MASK]")
        case_sensitive : bool
            Whether to perform case-sensitive matching (default: False)
        whole_words_only : bool
            Whether to match whole words only (default: True)
        **kwargs
            Additional parameters

        Returns
        -------
        List[Dict] or Dict
            Results with format:
            {
                "original_text": str,           # Original input text
                "masked_text": str,             # Text with entities masked
                "contains_entities": bool,      # True if entities found
                "entities": List[str]           # List of detected entities
            }

        Examples
        --------
        >>> result = generator.detect_and_mask_entities("I love my iPhone and Tesla", "brand names")
        >>> print(result)
        {
            "original_text": "I love my iPhone and Tesla",
            "masked_text": "I love my [MASK] and [MASK]",
            "contains_entities": True,
            "entities": ["iPhone", "Tesla"]
        }
        """
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False

        # Step 1: Detect entities
        detection_results = self.detect_entities(
            texts, entity_type, examples=examples, context=context,
            temperature=temperature, **kwargs
        )

        # Ensure detection_results is a list
        if isinstance(detection_results, dict):
            detection_results = [detection_results]

        # Step 2: Combine detection and masking
        final_results = []
        for result in detection_results:
            original_text = result["text"]
            contains_entities = result["contains_entities"]
            entities = result["entities"]

            # Apply masking if entities were found
            if contains_entities and entities:
                masked_text = self.mask_detected_entities(
                    original_text, entities, mask_token=mask_token,
                    case_sensitive=case_sensitive, whole_words_only=whole_words_only
                )
            else:
                masked_text = original_text

            # Create lean result
            final_result = {
                "original_text": original_text,
                "masked_text": masked_text,
                "contains_entities": contains_entities,
                "entities": entities
            }

            final_results.append(final_result)

        if return_single:
            return final_results[0] if final_results else {
                "original_text": texts[0] if texts else "",
                "masked_text": texts[0] if texts else "",
                "contains_entities": False,
                "entities": []
            }
        else:
            return final_results
