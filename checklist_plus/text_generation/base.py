import itertools

from pattern.en import wordnet

# Re-export all utility functions for backward compatibility
__all__ = [
    'all_synsets',
    'clean_senses',
    'all_possible_synonyms',
    'all_possible_antonyms',
    'all_possible_hypernyms',
    'all_possible_hyponyms',
    'all_possible_related',
]


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
