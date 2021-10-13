import six
import types
from difflib import SequenceMatcher

from . import backbone


def get_architectures():
    """
    get all of model architectures
    """
    names = []
    for k, v in backbone.__dict__.items():
        if isinstance(v, (types.FunctionType, six.class_types)):
            names.append(k)
    return names


def similar_architectures(name='', names=[], thresh=0.1, topk=10):
    """
    inferred similar architectures
    """
    scores = []
    for idx, n in enumerate(names):
        if n.startswith('__'):
            continue
        score = SequenceMatcher(None, n.lower(), name.lower()).quick_ratio()
        if score > thresh:
            scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    similar_names = [names[s[0]] for s in scores[:min(topk, len(scores))]]
    return similar_names
