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
