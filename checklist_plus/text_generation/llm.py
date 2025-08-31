import os
import re
from typing import Any, Dict, List, Optional, Union

from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from checklist_plus.config import cfg
from checklist_plus.text_generation.models import ParaphraseResponse, UniqueCompletions


class LLMTextGenerator:
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

    def unmask_multiple(self, texts, n_completions=1, candidates=None, metric='avg', context=None, **kwargs):
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
        if context:
            input_variables.append("context")
            input_data["context"] = context
        prompt_parts.append(cfg.config.text_generation.llm.unmask_prompt.task_context)
        if context:
            prompt_parts.append(cfg.config.text_generation.llm.unmask_prompt.background_data)
        prompt_parts.extend([cfg.config.text_generation.llm.unmask_prompt.rules,
                        cfg.config.text_generation.llm.unmask_prompt.task,
                        cfg.config.text_generation.llm.unmask_prompt.thinking_step,
                        cfg.config.text_generation.llm.unmask_prompt.output_format])
        prompt_text = "\n".join(prompt_parts)
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

                        # Score decreases with position (first suggestion gets highest score)
                        score = 1.0 - (i * 0.01)  # Smaller decrement for more granular scoring
                        all_results.append((completion_set, full_text, score))
                    else:
                        print(f"Warning: completion set {completion_set} has {len(completion_set)} items but expected {mask_count}")

        except Exception as e:
            print(f"LLM unmask failed for text '{text}': {e}")
            # Fallback: return original text with empty completions
            all_results.append(([""] * mask_count, text, 0.0))

        # print('all_results:', all_results)
        # print(f'Total unique completions generated: {len(unique_completions)}')
        return all_results

    def unmask(self, text_with_mask, n_completions=10, candidates=None):
        raise NotImplementedError("Use unmask_multiple for LLMTextGenerator")

    def replace_word(self, text, word, threshold=5, beam_size=100, candidates=None):
        """
        Replace a word in text using LLM.

        Parameters
        ----------
        text : str
            Original text
        word : str
            Word to replace
        threshold : float
            Threshold for replacement (not used in LLM version)
        beam_size : int
            Number of suggestions to generate
        candidates : List[str], optional
            Candidate words to consider

        Returns
        -------
        List[Tuple[List[str], str, float]]
            List of (replacement_words, full_text, score) tuples
        """
        # Replace the word with a blank
        masked_text = text.replace(word, "___")

        # Create a prompt template for word replacement
        replacement_template = PromptTemplate(
            input_variables=["beam_size", "masked_text"],
            template="""You are a word replacement expert. Replace the blank (___) with a different word that makes sense in the context.
Provide {beam_size} different replacements, each on a new line. Only provide the replacement word.

Text with blank: {masked_text}"""
        )

        # Format the prompt with variables
        formatted_prompt = replacement_template.format(
            beam_size=beam_size,
            masked_text=masked_text
        )

        try:
            response = self.llm_client.invoke(formatted_prompt)
            replacements = [line.strip() for line in response.content.strip().split('\n') if line.strip()]

            # Clean up replacements: remove numbering and extra formatting
            cleaned_replacements = []
            for replacement in replacements:
                # Remove numbering patterns like "1.", "2.", "1)", "2)", etc.
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', replacement.strip())
                # Remove any remaining numbers at the beginning
                cleaned = re.sub(r'^\d+\s*', '', cleaned)
                # Remove any trailing punctuation that might be artifacts
                cleaned = cleaned.strip(' .')
                if cleaned:  # Only add non-empty replacements
                    cleaned_replacements.append(cleaned)

            results = []
            for i, replacement in enumerate(cleaned_replacements[:beam_size]):
                if replacement.lower() != word.lower():  # Avoid replacing with the same word
                    full_text = text.replace(word, replacement)
                    score = 1.0 - (i * 0.1)
                    results.append(([replacement], full_text, score))

            return results

        except Exception as e:
            print(f"LLM replace_word failed for text '{text}', word '{word}': {e}")
            return []

    def antonyms(self, texts, word, threshold=5, pos=None, **kwargs):
        """
        Find antonyms using LLM.

        Parameters
        ----------
        texts : List[str] or str
            Context texts
        word : str
            Word to find antonyms for
        threshold : float
            Threshold for filtering (not used in LLM version)
        pos : str, optional
            Part of speech (not used in LLM version)
        **kwargs
            Additional parameters

        Returns
        -------
        List[Tuple[List[str], str, float]]
            List of (antonyms, context_text, score) tuples
        """
        if isinstance(texts, str):
            texts = [texts]

        # Create a prompt template for finding antonyms
        antonyms_template = PromptTemplate(
            input_variables=["word", "text"],
            template="""You are a word relationship expert. Find antonyms (opposites) of the given word that fit well in the provided context.
Provide antonyms that are natural and contextually appropriate.

Find antonyms of '{word}' that fit well in this context:

{text}"""
        )

        results = []
        for text in texts:
            # Format the prompt with variables
            formatted_prompt = antonyms_template.format(
                word=word,
                text=text
            )

            try:
                response = self.llm_client.invoke(formatted_prompt)
                antonyms = [word.strip() for word in response.content.strip().split('\n') if word.strip()]

                # Clean up antonyms: remove numbering and extra formatting
                cleaned_antonyms = []
                for antonym in antonyms:
                    # Remove numbering patterns like "1.", "2.", "1)", "2)", etc.
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', antonym.strip())
                    # Remove any remaining numbers at the beginning
                    cleaned = re.sub(r'^\d+\s*', '', cleaned)
                    # Remove any trailing punctuation that might be artifacts
                    cleaned = cleaned.strip(' .')
                    if cleaned:  # Only add non-empty antonyms
                        cleaned_antonyms.append(cleaned)

                for antonym in cleaned_antonyms[:5]:  # Limit to 5 antonyms
                    results.append(([antonym], text, 1.0))

            except Exception as e:
                print(f"LLM antonyms failed for word '{word}' in text '{text}': {e}")

        return results

    def synonyms(self, texts, word, threshold=5, pos=None, **kwargs):
        """
        Find synonyms using LLM.

        Parameters
        ----------
        texts : List[str] or str
            Context texts
        word : str
            Word to find synonyms for
        threshold : float
            Threshold for filtering (not used in LLM version)
        pos : str, optional
            Part of speech (not used in LLM version)
        **kwargs
            Additional parameters

        Returns
        -------
        List[Tuple[List[str], str, float]]
            List of (synonyms, context_text, score) tuples
        """
        if isinstance(texts, str):
            texts = [texts]

        # Create a prompt template for finding synonyms
        synonyms_template = PromptTemplate(
            input_variables=["word", "text"],
            template="""You are a word relationship expert. Find synonyms of the given word that fit well in the provided context.
Provide synonyms that are natural and contextually appropriate.

Find synonyms of '{word}' that fit well in this context:

{text}"""
        )

        results = []
        for text in texts:
            # Format the prompt with variables
            formatted_prompt = synonyms_template.format(
                word=word,
                text=text
            )

            try:
                response = self.llm_client.invoke(formatted_prompt)
                synonyms = [word.strip() for word in response.content.strip().split('\n') if word.strip()]

                # Clean up synonyms: remove numbering and extra formatting
                cleaned_synonyms = []
                for synonym in synonyms:
                    # Remove numbering patterns like "1.", "2.", "1)", "2)", etc.
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', synonym.strip())
                    # Remove any remaining numbers at the beginning
                    cleaned = re.sub(r'^\d+\s*', '', cleaned)
                    # Remove any trailing punctuation that might be artifacts
                    cleaned = cleaned.strip(' .')
                    if cleaned:  # Only add non-empty synonyms
                        cleaned_synonyms.append(cleaned)

                for synonym in cleaned_synonyms[:5]:  # Limit to 5 synonyms
                    results.append(([synonym], text, 1.0))

            except Exception as e:
                print(f"LLM synonyms failed for word '{word}' in text '{text}': {e}")

        return results

    def more_general(self, texts, word, threshold=5, pos=None, **kwargs):
        """Find more general terms (hypernyms) using LLM."""
        if isinstance(texts, str):
            texts = [texts]

        # Create a prompt template for finding more general terms
        general_template = PromptTemplate(
            input_variables=["word", "text"],
            template="""You are a word relationship expert. Find more general terms (hypernyms) of the given word that fit well in the provided context.

Find more general terms for '{word}' that fit well in this context:

{text}"""
        )

        results = []
        for text in texts:
            # Format the prompt with variables
            formatted_prompt = general_template.format(
                word=word,
                text=text
            )

            try:
                response = self.llm_client.invoke(formatted_prompt)
                hypernyms = [word.strip() for word in response.content.strip().split('\n') if word.strip()]

                # Clean up hypernyms: remove numbering and extra formatting
                cleaned_hypernyms = []
                for hypernym in hypernyms:
                    # Remove numbering patterns like "1.", "2.", "1)", "2)", etc.
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', hypernym.strip())
                    # Remove any remaining numbers at the beginning
                    cleaned = re.sub(r'^\d+\s*', '', cleaned)
                    # Remove any trailing punctuation that might be artifacts
                    cleaned = cleaned.strip(' .')
                    if cleaned:  # Only add non-empty hypernyms
                        cleaned_hypernyms.append(cleaned)

                for hypernym in cleaned_hypernyms[:5]:
                    results.append(([hypernym], text, 1.0))

            except Exception as e:
                print(f"LLM more_general failed for word '{word}' in text '{text}': {e}")

        return results

    def more_specific(self, texts, word, threshold=5, depth=3, pos=None, **kwargs):
        """Find more specific terms (hyponyms) using LLM."""
        if isinstance(texts, str):
            texts = [texts]

        system_prompt = """You are a word relationship expert. Find more specific terms (hyponyms) of the given word that fit well in the provided context."""

        results = []
        for text in texts:
            user_prompt = f"Find more specific terms for '{word}' that fit well in this context:\n\n{text}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            try:
                response = self.llm_client.invoke(messages)
                hyponyms = [word.strip() for word in response.content.strip().split('\n') if word.strip()]

                # Clean up hyponyms: remove numbering and extra formatting
                cleaned_hyponyms = []
                for hyponym in hyponyms:
                    # Remove numbering patterns like "1.", "2.", "1)", "2)", etc.
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', hyponym.strip())
                    # Remove any remaining numbers at the beginning
                    cleaned = re.sub(r'^\d+\s*', '', cleaned)
                    # Remove any trailing punctuation that might be artifacts
                    cleaned = cleaned.strip(' .')
                    if cleaned:  # Only add non-empty hyponyms
                        cleaned_hyponyms.append(cleaned)

                for hyponym in cleaned_hyponyms[:5]:
                    results.append(([hyponym], text, 1.0))

            except Exception as e:
                print(f"LLM more_specific failed for word '{word}' in text '{text}': {e}")

        return results

    def related_words(self, texts, words, threshold=5, depth=3, pos=None, **kwargs):
        """Find related words using LLM."""
        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(words, list):
            words = [words]

        system_prompt = """You are a word relationship expert. Find words that are related to the given word(s) that fit well in the provided context."""

        results = []
        for text in texts:
            word_list = ", ".join(words)
            user_prompt = f"Find words related to '{word_list}' that fit well in this context:\n\n{text}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            try:
                response = self.llm_client.invoke(messages)
                related = [word.strip() for word in response.content.strip().split('\n') if word.strip()]

                # Clean up related words: remove numbering and extra formatting
                cleaned_related = []
                for word in related:
                    # Remove numbering patterns like "1.", "2.", "1)", "2)", etc.
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', word.strip())
                    # Remove any remaining numbers at the beginning
                    cleaned = re.sub(r'^\d+\s*', '', cleaned)
                    # Remove any trailing punctuation that might be artifacts
                    cleaned = cleaned.strip(' .')
                    if cleaned:  # Only add non-empty words
                        cleaned_related.append(cleaned)

                for word in cleaned_related[:5]:
                    results.append(([word], text, 1.0))

            except Exception as e:
                print(f"LLM related_words failed for words '{words}' in text '{text}': {e}")

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

        # Create LLM client with specified temperature
        temp_llm = ChatOpenAI(
            openai_api_key=self.llm_client.openai_api_key,
            model_name=self.model_name,
            temperature=temperature
        )

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
            print(f"LLM batch paraphrase failed: {e}")
            raise e

        return all_paraphrases

    def filter_options(self, texts, word, options, threshold=5):
        """Filter options based on context using LLM."""
        if isinstance(texts, str):
            texts = [texts]

        system_prompt = """You are a text analysis expert. Given a list of options and a context, determine which options fit well in the context."""

        results = []
        for text in texts:
            options_list = ", ".join(options)
            user_prompt = f"Which of these options fit well in this context?\n\nContext: {text}\n\nOptions: {options_list}\n\nProvide only the options that fit well, one per line."

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            try:
                response = self.llm_client.invoke(messages)
                filtered = [word.strip() for word in response.content.strip().split('\n') if word.strip()]

                # Clean up filtered options: remove numbering and extra formatting
                cleaned_filtered = []
                for option in filtered:
                    # Remove numbering patterns like "1.", "2.", "1)", "2)", etc.
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', option.strip())
                    # Remove any remaining numbers at the beginning
                    cleaned = re.sub(r'^\d+\s*', '', cleaned)
                    # Remove any trailing punctuation that might be artifacts
                    cleaned = cleaned.strip(' .')
                    if cleaned:  # Only add non-empty options
                        cleaned_filtered.append(cleaned)

                for option in cleaned_filtered:
                    if option in options:
                        results.append(([option], text, 1.0))

            except Exception as e:
                print(f"LLM filter_options failed for word '{word}' in text '{text}': {e}")

        return results
