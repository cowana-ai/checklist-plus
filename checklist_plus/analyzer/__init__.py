"""
Text Analysis Module for CheckList Plus

This module provides spaCy-based text analysis with enriched Doc objects
that are compatible with standard nlp.pipe() but include additional metrics.
"""

from .base import BaseAnalyzer, TextAnalyzer, TextStatistics

__all__ = [
    'TextAnalyzer',
    'BaseAnalyzer',
    'TextStatistics'
]
