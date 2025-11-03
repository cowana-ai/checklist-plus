import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import spacy
from pydantic import BaseModel, Field
from spacy import tokens
from spacy.lang.en import English
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Configure logging
logger = logging.getLogger(__name__)


# Simple data structure for text statistics
class TextStatistics(BaseModel):
    """Basic text statistics"""
    char_count: int = Field(..., description="Total character count")
    word_count: int = Field(..., description="Total word count")
    sentence_count: int = Field(..., description="Total sentence count")
    paragraph_count: int = Field(..., description="Total paragraph count")
    avg_words_per_sentence: float = Field(..., description="Average words per sentence")
    avg_chars_per_word: float = Field(..., description="Average characters per word")
    unique_words: int = Field(..., description="Number of unique words")
    lexical_diversity: float = Field(..., description="Lexical diversity (unique/total words)")


class BaseAnalyzer(ABC):
    """Abstract base class for text analyzers"""

    @abstractmethod
    def analyze(self, text: str) -> dict[str, Any]:
        """Analyze a single text and return results"""
        pass

    @abstractmethod
    def analyze_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Analyze a batch of texts and return results"""
        pass


class TextAnalyzer:
    """
    Comprehensive text analyzer with multiple analysis capabilities.

    This class provides various text analysis methods including:
    - spaCy-based linguistic analysis
    - Basic text statistics
    - Extensible custom analysis pipeline
    """

    def __init__(self,
                 spacy_model: str = "en_core_web_sm",
                 enable_spacy: bool = True,
                 custom_analyzers: list[BaseAnalyzer] | None = None,
                 batch_size: int = 100,
                 hf_ner_model: str = "dslim/bert-base-NER-uncased"):
        """
        Initialize the TextAnalyzer.

        Args:
            spacy_model: Name of the spaCy model to load
            enable_spacy: Whether to enable spaCy-based analysis
            custom_analyzers: List of custom analyzer instances
            batch_size: Batch size for processing multiple texts
        """
        self.spacy_model_name = spacy_model
        self.enable_spacy = enable_spacy
        self.batch_size = batch_size
        self.custom_analyzers = custom_analyzers or []

        # Initialize spaCy
        self.nlp = None
        if self.enable_spacy:
            self._load_spacy_model()

        logger.info(f"TextAnalyzer initialized with spaCy: {self.enable_spacy}")

        tokenizer = AutoTokenizer.from_pretrained(hf_ner_model)
        model = AutoModelForTokenClassification.from_pretrained(hf_ner_model)

        self.hf_ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

    def _load_spacy_model(self):
        """Load the spaCy model with error handling"""
        try:
            self.nlp = spacy.load(self.spacy_model_name)
            logger.info(f"Loaded spaCy model: {self.spacy_model_name}")
        except OSError as e:
            logger.error(f"Could not load spaCy model '{self.spacy_model_name}': {e}")
            logger.info("Falling back to basic English tokenizer")
            try:
                self.nlp = English()
                self.nlp.add_pipe('sentencizer')
            except Exception as fallback_error:
                logger.error(f"Could not create fallback tokenizer: {fallback_error}")
                self.enable_spacy = False
                self.nlp = None

    def analyze_text_statistics(self, text: str) -> TextStatistics:
        """
        Analyze basic text statistics.

        Args:
            text: Input text to analyze

        Returns:
            TextStatistics object with various metrics
        """
        # Basic counts
        char_count = len(text)
        words = text.split()
        word_count = len(words)

        # Sentence count (simple heuristic)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        # Paragraph count
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        paragraph_count = len(paragraphs)

        # Averages
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        avg_chars_per_word = char_count / max(word_count, 1)

        # Lexical diversity
        unique_words = len({word.lower() for word in words})
        lexical_diversity = unique_words / max(word_count, 1)

        return TextStatistics(
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_words_per_sentence=avg_words_per_sentence,
            avg_chars_per_word=avg_chars_per_word,
            unique_words=unique_words,
            lexical_diversity=lexical_diversity
        )





    def analyze_readability(self, text: str) -> dict[str, float]:
        """
        Analyze text readability using various metrics.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with readability metrics
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not words or not sentences:
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'avg_sentence_length': 0.0,
                'avg_syllables_per_word': 0.0
            }

        # Calculate syllables (rough approximation)
        def count_syllables(word):
            word = word.lower()
            syllables = 0
            vowels = 'aeiouy'
            if word[0] in vowels:
                syllables += 1
            for i in range(1, len(word)):
                if word[i] in vowels and word[i-1] not in vowels:
                    syllables += 1
            if word.endswith('e'):
                syllables -= 1
            if syllables == 0:
                syllables = 1
            return syllables

        total_syllables = sum(count_syllables(word) for word in words)
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)

        # Flesch Reading Ease
        flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

        # Flesch-Kincaid Grade Level
        flesch_kincaid_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59

        return {
            'flesch_reading_ease': round(flesch_reading_ease, 2),
            'flesch_kincaid_grade': round(flesch_kincaid_grade, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_syllables_per_word': round(avg_syllables_per_word, 2)
        }

    def _match_hf_ner_to_tokens(self, doc, hf_ner_entities: list[dict[str, Any]]):
        """
        Match HF NER entities to spaCy tokens and add hf_ner attribute to each token.

        Args:
            doc: spaCy Doc object
            hf_ner_entities: List of HF NER entities with 'start', 'end' character positions
        """
        # Initialize hf_ner attribute for all tokens
        if not spacy.tokens.Token.has_extension("hf_ner"):
            spacy.tokens.Token.set_extension("hf_ner", default=None)

        # Create a mapping of character positions to entities
        char_to_entity = {}
        for entity in hf_ner_entities:
            start, end = entity['start'], entity['end']
            for char_pos in range(start, end):
                char_to_entity[char_pos] = entity

        # Match tokens to entities based on character overlap
        for token in doc:
            token_start = token.idx
            token_end = token.idx + len(token.text)

            # Find the best matching entity for this token
            best_entity = None
            max_overlap = 0

            for entity in hf_ner_entities:
                entity_start, entity_end = entity['start'], entity['end']

                # Calculate overlap between token and entity character spans
                overlap_start = max(token_start, entity_start)
                overlap_end = min(token_end, entity_end)
                overlap = max(0, overlap_end - overlap_start)

                # If this entity has more overlap with the token, use it
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_entity = entity

            # Assign the best matching entity to the token (or None if no good match)
            if max_overlap > 0:
                token._.hf_ner = best_entity
            else:
                token._.hf_ner = None

    def pipe_enriched(self, texts: list[str],
                     include_readability: bool = True,
                     include_statistics: bool = True,
                     include_hf_ner: bool = True,
                     batch_size: int | None = None) -> list[Any]:
        """
        Process texts using spaCy pipe and enrich Doc objects with additional analysis.
        Returns spaCy Doc objects with added custom attributes for richer analysis.

        This method is compatible with standard spaCy nlp.pipe() but provides enriched results:
        - Standard: parsed_data = list(nlp.pipe(test_queries))
        - Enriched: parsed_data = analyzer.pipe_enriched(test_queries)

        Args:
            texts: List of input texts to analyze
            include_readability: Whether to add readability metrics
            include_statistics: Whether to add text statistics
            include_hf_ner: Whether to add Hugging Face NER analysis
            batch_size: Batch size for processing (uses analyzer default if None)

        Returns:
            List of spaCy Doc objects with enriched custom attributes:
            - doc._.statistics: TextStatistics object
            - doc._.readability: Readability metrics dict
            - doc._.hf_ner: Hugging Face NER results list
            - doc._.custom_analyses: Dict of custom analysis results
            - token._.hf_ner: HF NER entity info for each token (when include_hf_ner=True)
        """
        if not self.enable_spacy or self.nlp is None:
            raise ValueError("spaCy is not available or enabled")

        # Use provided batch_size or fall back to analyzer's batch_size
        effective_batch_size = batch_size or self.batch_size

        try:
            # Process texts using spaCy pipe
            docs = list(self.nlp.pipe(texts, batch_size=effective_batch_size))

            # Batch process HF NER if enabled (more efficient than individual calls)
            hf_ner_results = []
            if include_hf_ner and hasattr(self, 'hf_ner_pipeline'):
                try:
                    # Process all texts at once for efficiency
                    hf_ner_results = self.hf_ner_pipeline(texts, batch_size=effective_batch_size)
                except Exception as e:
                    logger.error(f"Error in HF NER processing: {e}")
                    hf_ner_results = [[] for _ in texts]  # Empty results for all texts
            else:
                hf_ner_results = [[] for _ in texts]  # Empty results if disabled

            # Enrich each doc with additional analysis
            for i, doc in enumerate(docs):
                text = texts[i]

                # Add text statistics
                if include_statistics:
                    statistics = self.analyze_text_statistics(text)
                    # Set custom attribute on the doc
                    doc.set_extension("statistics", default=None, force=True)
                    doc._.statistics = statistics

                # Add readability metrics
                if include_readability:
                    readability = self.analyze_readability(text)
                    doc.set_extension("readability", default=None, force=True)
                    doc._.readability = readability

                # Add HF NER results
                if include_hf_ner:
                    doc.set_extension("hf_ner", default=None, force=True)
                    doc._.hf_ner = hf_ner_results[i]

                    # Match HF NER entities to individual tokens
                    self._match_hf_ner_to_tokens(doc, hf_ner_results[i])

                # Add custom analyses
                custom_analyses = {}
                for analyzer in self.custom_analyzers:
                    try:
                        analyzer_name = analyzer.__class__.__name__
                        if hasattr(analyzer, 'analyze_batch'):
                            # Handle batch analyzers separately after the loop
                            continue
                        else:
                            custom_analyses[analyzer_name] = analyzer.analyze(text)
                    except Exception as e:
                        logger.error(f"Error in custom analyzer {analyzer.__class__.__name__}: {e}")

                doc.set_extension("custom_analyses", default=None, force=True)
                doc._.custom_analyses = custom_analyses

            # Handle batch-capable custom analyzers
            for analyzer in self.custom_analyzers:
                if hasattr(analyzer, 'analyze_batch'):
                    try:
                        analyzer_name = analyzer.__class__.__name__
                        batch_results = analyzer.analyze_batch(texts)
                        for i, doc in enumerate(docs):
                            if i < len(batch_results):
                                if not hasattr(doc._, 'custom_analyses') or doc._.custom_analyses is None:
                                    doc._.custom_analyses = {}
                                doc._.custom_analyses[analyzer_name] = batch_results[i]
                    except Exception as e:
                        logger.error(f"Error in batch custom analyzer {analyzer.__class__.__name__}: {e}")

            return docs

        except Exception as e:
            logger.error(f"Error in pipe_enriched processing: {e}")
            raise



    def add_custom_analyzer(self, analyzer: BaseAnalyzer):
        """
        Add a custom analyzer to the analysis pipeline.

        Args:
            analyzer: Instance of BaseAnalyzer to add
        """
        self.custom_analyzers.append(analyzer)
