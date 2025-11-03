"""E-commerce specific perturbations"""
import re

import numpy as np

from checklist_plus.perturb.base import Perturb, process_ret


class EcommercePerturb(Perturb):
    """E-commerce domain specific perturbations for product names, models, prices, etc."""
    @staticmethod
    def add_product_model_variations(doc, meta=False, seed=None, n=10):
        """Add variations for product model formatting using spaCy analytics

        Uses linguistic analysis to detect product variants without hardcoding:
        - "iPhone 15 Pro" -> "iPhone15Pro", "iPhone-15-Pro"
        - "MacBook Air" -> "MacBookAir", "MacBook-Air"
        - "Galaxy S24 Ultra" -> "GalaxyS24Ultra", "Galaxy-S24-Ultra"

        Parameters
        ----------
        doc : spacy.token.Doc
            input (parsed with spaCy)
        meta : bool
            if True, will return list of (original_phrase, changed_phrase) as meta
        seed : int
            random seed
        n : int
            number of variations to generate

        Returns
        -------
        list(str)
            if meta=True, returns (list(str), list(tuple))
            Strings with product model variations
        """
        if seed is not None:
            np.random.seed(seed)

        ret = []
        ret_m = []
        text = doc.text

        def is_short_adjective_like(word_token, len_threshold=7):
            """Check if word looks like a short product variant using spaCy analysis"""
            # Find the token in the spaCy doc
            if word_token is None:
                return False
            if len(word_token.text) > len_threshold:
                return False
            # Rule 1: Use spaCy's POS tagging - ADJ, PROPN, or NOUN that look like variants
            if word_token.pos_ in ['ADJ', 'PROPN']:
                return True

            # Rule 2: spaCy detects it as unknown/foreign (X tag) but looks product-like
            if word_token.pos_ == 'X' and word_token.text[0].isupper():
                return True

            return False

        # Basic patterns with boundary detection
        patterns = [
            # # Brand + Model Number (iPhone 15, Galaxy S24)
            # (r'\b([A-Z][a-z]{2,})\s+([A-Z]?\d+[A-Za-z]*)\b(?!\s+[a-z])', r'\1\2'),
            # (r'\b([A-Z][a-z]{2,})\s+([A-Z]?\d+[A-Za-z]*)\b(?!\s+[a-z])', r'\1-\2'),

            # # Brand + Single Letter Model (Model X, Series S)
            # (r'\b([A-Z][a-z]{2,})\s+([A-Z])\b(?!\s+[a-z])', r'\1\2'),
            # (r'\b([A-Z][a-z]{2,})\s+([A-Z])\b(?!\s+[a-z])', r'\1-\2'),

            # # Software versions (iOS 17.1, Windows 11)
            # (r'\b([A-Za-z]{2,})\s+(\d+(?:\.\d+)*)\b(?!\s+[a-z])', r'\1\2'),
            # (r'\b([A-Za-z]{2,})\s+(\d+(?:\.\d+)*)\b(?!\s+[a-z])', r'\1-\2'),
        ]

        # Dynamic patterns using spaCy analysis
        tokens = [token for token in doc if not token.is_space]

        for i in range(len(tokens) - 1):
            brand_token = tokens[i]
            next_token = tokens[i + 1]

            # Check if this looks like Brand + Product Variant or Brand + Number using spaCy
            is_brand_or_product_like = (brand_token._.hf_ner is not None and ("ORG" in brand_token._.hf_ner['entity'] or "MISC" in brand_token._.hf_ner['entity']) and brand_token._.hf_ner['score'] > 0.5) or (brand_token.pos_ in ['PROPN', 'NOUN'])

            is_variant_like = is_short_adjective_like(next_token) or (next_token.pos_ == 'NUM' or next_token.text.isdigit())
            if is_brand_or_product_like and is_variant_like:
                # Create patterns for this specific combination
                brand_escaped = re.escape(brand_token.text)
                variant_escaped = re.escape(next_token.text)

                # Simple pattern without negative lookahead first
                simple_pattern = rf'\b{brand_escaped}\s+{variant_escaped}\b'

                patterns.extend([
                    (simple_pattern, f'{brand_token.text}{next_token.text}'),
                    (simple_pattern, f'{brand_token.text}-{next_token.text}')
                ])
        # Look for three-word combinations: Brand + Number + Variant or Brand + Variant + Number
        for i in range(len(tokens) - 2):
            brand_token = tokens[i]
            next_token = tokens[i + 1]
            next_next_token = tokens[i + 2]
            is_brand_or_product_like = (brand_token._.hf_ner is not None and ("ORG" in brand_token._.hf_ner['entity'] or "MISC" in brand_token._.hf_ner['entity']) and brand_token._.hf_ner['score'] > 0.5) or (brand_token.pos_ in ['PROPN', 'NOUN'])

            is_number = (next_token.pos_ == 'NUM' or next_token.text.isdigit()) or (next_next_token.pos_ == 'NUM' or next_next_token.text.isdigit())
            is_variant_like = is_short_adjective_like(next_next_token) or is_short_adjective_like(next_token)

            if is_brand_or_product_like and is_number and is_variant_like:
                brand_escaped = re.escape(brand_token.text)
                next_token_escaped = re.escape(next_token.text)
                next_next_token_escaped = re.escape(next_next_token.text)

                # Simple pattern without negative lookahead
                simple_pattern_three = rf'\b{brand_escaped}\s+{next_token_escaped}\s+{next_next_token_escaped}\b'

                patterns.extend([
                    (simple_pattern_three, f'{brand_token.text}{next_token.text}{next_next_token.text}'),
                    (simple_pattern_three, f'{brand_token.text} {next_next_token.text} {next_token.text}'),
                    (simple_pattern_three, f'{brand_token.text}-{next_token.text}-{next_next_token.text}')
                ])

        # Reverse patterns (compound -> spaced)
        # These patterns detect existing compound/hyphenated forms and convert them back to spaced
        all_reverse_patterns = [
            # iPhone15 -> iPhone 15, GalaxyS24 -> Galaxy S24
            (r'\b([A-Z][a-z]{2,})([A-Z]?\d+[A-Za-z]*)\b', r'\1 \2'),
            # iPhone-15 -> iPhone 15, Galaxy-S24 -> Galaxy S24
            (r'\b([A-Z][a-z]{2,})-([A-Z]?\d+[A-Za-z]*)\b', r'\1 \2'),
            # ModelX -> Model X, SeriesS -> Series S
            (r'\b([A-Z][a-z]{2,})([A-Z])\b', r'\1 \2'),
            # iOS17.1 -> iOS 17.1, Windows11 -> Windows 11
            (r'\b([A-Za-z]{2,})(\d+(?:\.\d+)*)\b', r'\1 \2'),
            # iOS-17 -> iOS 17, Windows-11 -> Windows 11
            (r'\b([A-Za-z]{2,})-(\d+(?:\.\d+)*)\b', r'\1 \2'),
        ]

        # Filter reverse patterns to only include those that match the current text
        reverse_patterns = []
        for pattern, replacement in all_reverse_patterns:
            if re.search(pattern, text):
                reverse_patterns.append((pattern, replacement))

        # Combine all patterns
        all_patterns = patterns + reverse_patterns
        print(all_patterns)
        variations_generated = 0

        # Try to generate variations
        for _ in range(n * 3):  # More attempts since patterns are more restrictive
            if variations_generated >= n:
                break

            pattern, replacement = all_patterns[np.random.choice(len(all_patterns))]

            # Check if pattern matches
            match = re.search(pattern, text)
            if match:
                # Apply transformation
                new_text = re.sub(pattern, replacement, text, count=1)

                if new_text != text and new_text not in ret:
                    # Extract what changed for metadata
                    original_phrase = match.group(0)
                    new_phrase = re.sub(pattern, replacement, original_phrase)

                    ret.append(new_text)
                    ret_m.append((original_phrase, new_phrase))
                    variations_generated += 1

        return process_ret(ret, ret_m=ret_m, n=n, meta=meta) if ret else None
