"""
Perturbation modules for CheckList Plus.

This package provides various text perturbation methods for behavioral testing,
including both rule-based and LLM-enhanced approaches.
"""

from checklist_plus.perturb.base import Perturb
from checklist_plus.perturb.ecommerce import EcommercePerturb
from checklist_plus.perturb.llm import LLMPerturb
from checklist_plus.utils import is_brand_fuzzy_match

__all__ = [
    'Perturb',
    'LLMPerturb',
    'EcommercePerturb',
]
