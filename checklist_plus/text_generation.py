import collections
import itertools
import json

# Add LLM imports
import os
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
import torch
import torch.nn.functional as F
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pattern.en import pluralize, wordnet
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from checklist_plus.config import cfg


class UniqueCompletions(BaseModel):
    """Pydantic model for generating unique text completions."""
    completions: list[list[str]] = Field(
        description="A list of completion sets. For single mask texts, each inner list has one completion. For multiple mask texts, each inner list contains completions in order for each mask position. Each completion should be either: 1) A single word, OR 2) A possessive noun phrase like 'goalkeeper's performance', 'student's homework', 'company's profits'. Avoid longer phrases like 'museum that I visited' or 'beautiful sunset today'. Example: [['game', 'it'], ['performance', 'food'], ['movie', 'service']] for two masks."
    )


def all_synsets(word, pos=None):
    map = {
        'NOUN': wordnet.NOUN,
        'VERB': wordnet.VERB,
        'ADJ': wordnet.ADJECTIVE,
        'ADV': wordnet.ADVERB
        }
    if pos is None:
        pos_list = [wordnet.VERB, wordnet.ADJECTIVE, wordnet.NOUN, wordnet.ADVERB]
    else:
        pos_list = [map[pos]]
    ret = []
    for pos in pos_list:
        ret.extend(wordnet.synsets(word, pos=pos))
    return ret

def clean_senses(synsets):
    return [x for x in set(synsets) if '_' not in x]
def all_possible_synonyms(word, pos=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        # if syn.synonyms[0] != word:
        #     continue
        ret.extend(syn.senses)
    return clean_senses(ret)

def all_possible_antonyms(word, pos=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        if not syn.antonym:
            continue
        for s in syn.antonym:
            ret.extend(s.senses)
    return clean_senses(ret)

def all_possible_hypernyms(word, pos=None, depth=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        ret.extend([y for x in syn.hypernyms(recursive=True, depth=depth) for y in x.senses])
    return clean_senses(ret)
def all_possible_hyponyms(word, pos=None, depth=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        ret.extend([y for x in syn.hyponyms(recursive=True, depth=depth) for y in x.senses])
    return clean_senses(ret)
def all_possible_related(words, pos=None, depth=1):
    all_syns = [y for word in words for y in all_synsets(word, pos=pos)]
    # all_syns = [all_synsets(x, pos=pos) for x in words]
    # all_syns = [x[0] for x in all_syns if x]
    # return all_syns
    # print(all_syns)
    all_ancestors = [wordnet.ancestor(s1, s2) for s1, s2 in itertools.combinations(all_syns, 2)]
    all_ancestors = [x for x in all_ancestors if x]
    # print(all_ancestors)
    mapz = {x.lexname: x for x in all_ancestors}
    all_ancestors = list(mapz.values())
    all_descendents = [y for x in all_ancestors for y in x.hyponyms(recursive=True, depth=depth)]
    ret = [y for x in all_descendents for y in x.senses]
    return clean_senses(ret)

class TextGenerator:
    def __init__(self, url=None, model_name='roberta-base', prefix_sentence='', allow_word_pieces=False, **kwargs):
        self.url = url
        if url is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            # self.model = BertForMaskedLM.from_pretrained('bert-base-cased')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.prefix_sentence = prefix_sentence
            self.prefix_len = len(self.tokenizer.encode(prefix_sentence, add_special_tokens=False))
            self.allow_word_pieces = allow_word_pieces
            tmp = self.tokenizer.tokenize(' a')[0].split('a')
            assert len(tmp) == 2
            assert tmp[1] == ''
            self.space_prefix = tmp[0]
            if not self.allow_word_pieces:
                self.with_space = torch.tensor(np.array(list({i for x, i in self.tokenizer.get_vocab().items() if x.startswith(self.space_prefix)})), device=self.device);
                self.with_space_set = set(self.with_space.cpu().numpy())
                self.special_chars = {i for x, i in self.tokenizer.get_vocab().items() if not x.strip(self.space_prefix).isalnum()}

    def unmask_multiple(self, texts, beam_size=500, candidates=None, metric='avg', **kwargs):
        rets = []
        for text in texts:
            # print('text:', text)
            # print('candidates:', candidates)
            # print(len(self.unmask(text, beam_size, candidates)))
            rets.append(self.unmask(text, beam_size, candidates))
        scores = collections.defaultdict(float) if metric == 'avg' else collections.defaultdict(lambda: 999999999)
        count = collections.defaultdict(float)
        examples = {}
        longest = max([len(x[0][0]) for x in rets])
        rets = sorted(rets, key=lambda x:len(x[0][0]), reverse=True)
        # print('rets:', rets)
        for r in rets:
            for x in r:
                tup = tuple(x[0])
                if len(tup) != longest:
                    tups = [k for k in scores if tuple(k[:len(tup)]) == tup]
                else:
                    tups = [tup]
                for tup in tups:
                    count[tup] += 1
                    examples[tup] = x[1]
                    if metric == 'avg':
                        scores[tup] += x[-1]
                    elif metric == 'min':
                        scores[tup] = min(scores[tup], x[-1])
        if metric == 'min':
            for x in count:
                # print(x, count[x])
                if count[x] != len(texts):
                    scores[x] = -999999
        else:
            for x in scores:
                scores[x] = scores[x] / len(texts)
        scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        # print([(list(x[0]), examples[x[0]], x[1]) for x in scores])
        return [(list(x[0]), examples[x[0]], x[1]) for x in scores]



    def unmask(self, text_with_mask, beam_size=10, candidates=None):
        if self.url is not None:
            params = {'text': text_with_mask, 'beam_size': beam_size, 'candidates': candidates}
            r = requests.post(url='%s/unmask' % self.url, data={'params': json.dumps(params)})
            r = [tuple(x) for x in json.loads(r.text)]
            return r
        # print(text_with_mask)
        tokenizer = self.tokenizer
        model = self.model
        encoded = np.array(tokenizer.encode(self.prefix_sentence + text_with_mask, add_special_tokens=True))
        cands = []
        if candidates is not None:
            candidates = candidates + [self.space_prefix + x for x in candidates]
            cands = tokenizer.convert_tokens_to_ids(candidates)
            if self.allow_word_pieces:
                cands_with_space = list(set(cands))
            else:
                cands_with_space = list(set(cands).intersection(self.with_space_set))
            if not len(cands_with_space):
                return []
        input_ids = torch.tensor(encoded)
        # toks = tokenizer.tokenize('[CLS] %s [SEP]' % string)
        current_beam= [([], 0)]
        masked = (input_ids == self.tokenizer.mask_token_id).numpy().nonzero()[0]
        # print(masked)
        while len(current_beam[0][0]) != masked.shape[0]:
            current_beam = current_beam[:beam_size]
            size = len(current_beam[0][0])
            to_pred = []
            new_beam = []
            for i, current in enumerate(current_beam):
                idxs = current[0]
                c = encoded.copy()
                c[masked[:len(idxs)]] = idxs
                to_pred.append(c)
            # print('ae')
            # print('\n'.join([tokenizer.decode(x) for x in to_pred]))
            # print()
            to_pred = torch.tensor(to_pred, device=self.device).to(torch.int64)
            with torch.no_grad():
                outputs = model(to_pred)[0]
            for i, current in enumerate(current_beam):
                prev = int(to_pred[i][masked[size] - 1])
                forbid = False
                # allow tokens that don't start with space if previous is not alphanumeric
                if not self.allow_word_pieces and prev not in self.special_chars:
                    forbid = True
                    # print('Forbid Prev, current', prev,  tokenizer.decode(to_pred[i][masked[size] - 1:masked[size]+1]))
                if candidates is not None:
                    cands_to_use = cands_with_space if forbid else cands
                    scores = [outputs[i, masked[size], j] for j in cands_to_use]
                    new = [(current[0] + [int(x[0])], float(x[1]) + current[1]) for x in zip(cands_to_use, scores)]
                else:
                    if forbid:
                        v, top_preds = torch.topk(outputs[i, masked[size], self.with_space.to(torch.int64)], beam_size + 10)
                        top_preds = self.with_space[top_preds]
                    else:
                        v, top_preds = torch.topk(outputs[i, masked[size]], beam_size + 10)
                    new = [(current[0] + [int(x[0])], float(x[1]) + current[1]) for x in zip(top_preds, v)]
                new_beam.extend(new)
            current_beam = sorted(new_beam, key=lambda x:x[1], reverse=True)
        ret = []
        ret_text = []
        cop = encoded.copy()
        for idxs, score in current_beam:
            # words = tokenizer.convert_ids_to_tokens(idxs)
            words = [str(tokenizer.decode([i])).strip() for i in idxs]
            cop[masked] = idxs
            text = tokenizer.decode(cop[1 + self.prefix_len:-1])
            ret.append((words, text, score / masked.shape[0]))
        ret = sorted(ret, key=lambda x:x[2], reverse=True)
        # print('ret:', ret)
        return ret
    def fill_in_between(self, pieces, beam_size=10, candidates=None):
        text = ''
        for p in pieces[:-1]:
            text += p
            text += ' ' + self.tokenizer.mask_token
            if p != '':
                text += ' '
        text += pieces[-1]
        if pieces[-1] == '':
            text = text.rstrip()
        return self.unmask(text, beam_size=beam_size, candidates=candidates)

    def replace_word(self, text, word,  threshold=5, beam_size=100, candidates=None):
        masked = re.sub(r'\b%s\b' % re.escape(word), self.tokenizer.mask_token, text)
        if masked == text:
            return []
        if candidates is not None:
            candidates = [word] + candidates
        ret =  self.unmask(masked, beam_size=beam_size, candidates=candidates)
        non_word = [x for x in ret if np.all([y not in [self.tokenizer.unk_token, word] for y in x[0]])]
        score = [x for x in ret if np.all([y in [word, self.tokenizer.unk_token] for y in x[0]])]
        if not score:
            score = 0
        else:
            score = score[0][-1]
        escaped = re.escape(word)
        # new_ret = [(x[0], x[1], score - x[2]) for x in non_word if score - x[2] < threshold]
        try:
            new_ret = [(x[0], re.sub(r'\b%s\b' % escaped, x[0][0], text), score - x[2]) for x in non_word if score - x[2] < threshold]
        except:
            new_ret = [(x[0], x[1], score - x[2]) for x in non_word if score - x[2] < threshold]
        return new_ret

    def more_general(self, texts, word, threshold=5, pos=None, **kwargs):
        options = all_possible_hypernyms(word, pos=pos)
        # print(options)
        return self.filter_options(texts, word, options, threshold)
    def more_specific(self, texts, word, threshold=5, depth=3, pos=None, **kwargs):
        options = all_possible_hyponyms(word, depth=depth, pos=pos)
        return self.filter_options(texts, word, options, threshold)
    def related_words(self, texts, words, threshold=5, depth=3, pos=None, **kwargs):
        if type(words) != list:
            words = [words]
        if len(words) == 1:
            options = all_possible_hypernyms(words[0], pos=pos)
            ancestors = [x[0][0] for x in self.filter_options(texts, words[0], options, threshold)]
            # print(ancestors)
            options = list({y for x in ancestors for y in all_possible_hyponyms(x, depth=depth)})
        else:
            options = all_possible_related(words, depth=depth)
        return self.filter_options(texts, words[0], options, threshold)
    def antonyms(self, texts, word, threshold=5, pos=None, **kwargs):
        options = all_possible_antonyms(word, pos=pos)
        return self.filter_options(texts, word, options, threshold)
    def synonyms(self, texts, word, threshold=5, pos=None, **kwargs):
        options = all_possible_synonyms(word, pos=pos)
        # print(options)
        return self.filter_options(texts, word, options, threshold)

    def filter_options(self, texts, word, options, threshold=5):
        if type(texts) != list:
            texts = [texts]
        options = options + [word]
        in_all = set(options)
        orig_ret = []
        for text in texts:
            masked = re.sub(r'\b%s\b' % re.escape(word), self.tokenizer.mask_token, text)
            if masked == text:
                continue
            ret =  self.unmask(masked, beam_size=100, candidates=options)
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
        return [x for x in orig_ret if x[0][0] in in_all]

    def antonym(self, text, word, threshold=5, synonym=False):
        options = all_possible_antonyms(word)
        if synonym:
            options = all_possible_synonyms(word)
        if not options:
            return []
        options = options + [word]
        masked = re.sub(r'\b%s\b' % re.escape(word), '[MASK]', text)
        if masked == text:
            return []
        ret =  self.unmask(masked, beam_size=100000000, candidates=options)
        non_word = [x for x in ret if np.all([y not in [self.tokenizer.unk_token, word] for y in x[0]])]
        score = [x for x in ret if np.all([y in [word, self.tokenizer.unk_token] for y in x[0]])][0][-1]
        new_ret = [(x[0], x[1], score - x[2]) for x in non_word if score - x[2] < threshold]
        return new_ret
    def try_all_antonyms(self, text, threshold=5, synonym=False):
        if self.url is not None:
            params = {'text': text }
            r = requests.post(url='%s/tokenize' % self.url, data={'params': json.dumps(params)})
            words = json.loads(r.text)
        else:
            words = self.tokenizer.tokenize(text)
        new_ret = []
        for word in words:
            word = word.strip(self.space_prefix)
            try:
                if synonym:
                    ret = self.synonyms(text, word, threshold)
                else:
                    ret = self.antonyms(text, word, threshold)
            except:
                print('Error', word)
                print()
                continue
            new_ret.extend(ret)
        return sorted(new_ret, key=lambda x:x[2])

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

        for text in texts:
            # Count the number of masks in the text
            mask_count = text.count(self.tokenizer.mask_token)

            if mask_count == 0:
                continue

            # Replace [MASK] with a more LLM-friendly placeholder
            llm_text = text.replace(self.tokenizer.mask_token, "___")

            try:
                # Create context-aware prompt template (general for any number of masks)
                if context:
                    prompt_text = f"{cfg.config.text_generation.llm.unmask_prompt.task_context}\n{cfg.config.text_generation.llm.unmask_prompt.background_data}\n{cfg.config.text_generation.llm.unmask_prompt.rules}\n{cfg.config.text_generation.llm.unmask_prompt.task}\n{cfg.config.text_generation.llm.unmask_prompt.output_format}"
                    completion_template = PromptTemplate(
                        input_variables=["n_completions", "llm_text", "context", "mask_count"],
                        template=prompt_text
                    )

                    # Format the prompt with context
                    formatted_prompt = completion_template.format(
                        n_completions=n_completions,
                        llm_text=llm_text,
                        context=context,
                        mask_count=mask_count
                    )
                else:
                    prompt_text = f"{cfg.config.text_generation.llm.unmask_prompt.task_context}\n{cfg.config.text_generation.llm.unmask_prompt.rules}\n{cfg.config.text_generation.llm.unmask_prompt.task}\n{cfg.config.text_generation.llm.unmask_prompt.output_format}"
                    completion_template = PromptTemplate(
                        input_variables=["n_completions", "llm_text", "mask_count"],
                        template=prompt_text
                    )

                    # Format the prompt without context (diverse topics)
                    formatted_prompt = completion_template.format(
                        n_completions=n_completions,
                        llm_text=llm_text,
                        mask_count=mask_count
                    )

                # print("formatted_prompt:", formatted_prompt)

                # Use structured output with Pydantic model
                structured_llm = self.llm_client.with_structured_output(UniqueCompletions)
                # print("here")
                response = structured_llm.invoke(formatted_prompt)
                # print("response:", response)

                # Extract completions from Pydantic model
                completions = response.completions if hasattr(response, 'completions') else []

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
