"""
Utility functions for perturbation operations.
"""

def is_brand_fuzzy_match(word, similarity_threshold=0.85):
    """
    Check if a word is likely a brand name using fuzzy matching.

    Parameters
    ----------
    word : str
        Word to check
    similarity_threshold : float
        Similarity threshold for fuzzy matching (0.0-1.0)

    Returns
    -------
    bool
        True if word is likely a brand name
    """
    # Hardcoded list of problematic brand names for inflection
    # These are brands that would be ruined by plural/singular transformation
    # Focus on brands that look like regular nouns but are actually company names
    tricky_brand_names = {
        # Tech brands that look like nouns
        'apple', 'blackberry', 'mint', 'oracle', 'tesla', 'uber', 'zoom', 'slack', 'discord',
        'dropbox', 'box', 'square', 'stripe', 'coinbase', 'shopify', 'spotify', 'twitch',
        'steam', 'epic', 'valve', 'roku', 'nest', 'ring', 'fitbit', 'garmin', 'peloton',

        # Fashion/Sports brands that look like nouns
        'puma', 'jaguar', 'converse', 'vans', 'gap', 'target', 'champion', 'under', 'armour',
        'lululemon', 'patagonia', 'north', 'face', 'timberland', 'fossil', 'guess', 'diesel',

        # Auto brands that look like nouns
        'ford', 'dodge', 'ram', 'jeep', 'saturn', 'mercury', 'genesis', 'infinity', 'acura',
        'smart', 'mini', 'fiat', 'seat', 'skoda', 'dacia', 'alpine', 'lotus', 'mclaren',

        # Food/Beverage brands that look like nouns
        'subway', 'dominos', 'papa', 'johns', 'dairy', 'queen', 'burger', 'king', 'white',
        'castle', 'five', 'guys', 'shake', 'shack', 'chipotle', 'panera', 'bread', 'olive',
        'garden', 'red', 'lobster', 'outback', 'steakhouse', 'applebees', 'fridays', 'ruby',
        'tuesday', 'buffalo', 'wild', 'wings', 'hooters', 'dennys', 'ihop', 'waffle', 'house',

        # Retail brands that look like nouns
        'target', 'gap', 'old', 'navy', 'banana', 'republic', 'american', 'eagle', 'express',
        'forever', 'urban', 'outfitters', 'anthropologie', 'free', 'people', 'pacsun', 'zumiez',
        'foot', 'locker', 'finish', 'line', 'dick', 'sporting', 'goods', 'bass', 'pro', 'shops',

        # Financial brands that look like nouns
        'mint', 'robinhood', 'acorns', 'stash', 'wealthfront', 'betterment', 'sofi', 'lending',
        'tree', 'rocket', 'mortgage', 'quicken', 'loans', 'capital', 'one', 'discover', 'ally',

        # Airlines that look like nouns
        'spirit', 'frontier', 'allegiant', 'sun', 'country', 'virgin', 'atlantic', 'jetblue',
        'southwest', 'delta', 'united', 'american', 'alaska', 'hawaiian', 'breeze', 'avelo',

        # Media/Entertainment brands that look like nouns
        'hulu', 'paramount', 'peacock', 'discovery', 'lifetime', 'bravo', 'spike', 'comedy',
        'central', 'cartoon', 'network', 'adult', 'swim', 'toonami', 'boomerang', 'nickelodeon',

        # Gaming brands that look like nouns
        'steam', 'epic', 'origin', 'battlenet', 'uplay', 'gog', 'humble', 'bundle', 'itch',
        'twitch', 'discord', 'teamspeak', 'mumble', 'ventrilo', 'curse', 'overwolf', 'razer',

        # Health/Beauty brands that look like nouns
        'dove', 'ivory', 'tide', 'gain', 'dawn', 'joy', 'cheer', 'bounce', 'downy', 'febreze',
        'pledge', 'windex', 'lysol', 'clorox', 'ajax', 'comet', 'soft', 'scrub', 'magic', 'eraser',

    }

    # Convert word to lowercase for comparison
    word_lower = word.lower().strip()

    # Exact match check
    if word_lower in tricky_brand_names:
        return True

    # Fuzzy matching using simple string similarity
    from difflib import SequenceMatcher
    for brand in tricky_brand_names:
        similarity = SequenceMatcher(None, word_lower, brand).ratio()
        if similarity >= similarity_threshold:
            return True

    return False


def is_valid_noun(word: str, lang="eng"):
    """
    Check if a word exists in WordNet vocabulary.

    Parameters
    ----------
    word : str
        Word to check
    nlp_doc : spacy.token.Doc, optional
        spaCy document (kept for compatibility but not used)

    Returns
    -------
    bool
        True if word exists in WordNet vocabulary
    """
    if len(word) < 2 or not word.isalpha():
        return False


    import nltk
    from nltk.corpus import wordnet

    # Ensure wordnet is downloaded (this is safe to call multiple times)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

    # Check if word has any synsets in WordNet (any part of speech)
    word_lower = word.lower()
    if wordnet.synsets(word_lower, pos=wordnet.NOUN, lang=lang):
        return True

    return False
