# tokenizer.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from typing import Dict, Tuple, List
import util

from tokenizers import Tokenizer
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.decoders


def get_gpt2_tokenizer() -> Tokenizer:
    """
    Return a GPT-2 tokenizer.
    """
    vocab, merges = tokenizers.models.BPE.read_file("data/vocab.json", "data/merges.txt")
    clean_vocab(vocab, merges)
    tokenizer = Tokenizer(tokenizers.models.BPE(vocab, merges))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space = False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()

    return tokenizer


def clean_vocab(vocab: Dict[str, int], merges: List[Tuple[str, str]]):
    """
    Question:
        Given the vocabulary and merges of a BPE tokenizer, clean them up to avoid subtokens
        that consist of multiple digits. This would reduce the sparsity problem.

        This function does in-place modifications, so it should not return anything.

    Example:
        >>> vocab = {'Ġ': 0, '1': 1, '2': 2, 'Ġ1': 3, 'Ġ2': 4, '12': 5, 'Ġ12': 6}
        >>> merges = [('Ġ', '1'), ('Ġ', '2'), ('1', '2'), ('Ġ1', '2')]
        >>> clean_vocab(vocab, merges)
        >>> vocab
        {'Ġ': 0, '1': 1, '2': 2, 'Ġ1': 3, 'Ġ2': 4}

    Args:
        vocab (:obj:`Dict[str, int]`):
            A dictionary of string keys and their ids, e.g. `{"am": 0,...}`

        merges (:obj:`List[Tuple[str, str]]`):
            A list of pairs of tokens (:obj:`Tuple[str, str]`), e.g. `[("a", "b"),...]`
    """

    def is_multi_digit(token: str) -> bool:
        for i in range(len(token)):
            if token[i].isdigit():
                if i + 1 < len(token) and token[i + 1].isdigit():
                    return True
        return False
    
    tokens_to_remove = []
    for tok in vocab.keys():
        if is_multi_digit(tok):
            tokens_to_remove.append(tok)
    
    for tok in tokens_to_remove:
        vocab.pop(tok)
    
    sorted_tokens = sorted(vocab.keys(), key=lambda t: vocab[t])
    
    final_cleaned_vocab = {}
    for new_id, tok in enumerate(sorted_tokens):
        final_cleaned_vocab[tok] = new_id
    
    vocab.clear()
    vocab.update(final_cleaned_vocab)

    cleaned_merges = []
    for a, b in merges:
        merged_token = a + b
        if not is_multi_digit(merged_token):
            if a in vocab and b in vocab:
                cleaned_merges.append((a,b))

    merges[:] = cleaned_merges


if __name__ == '__main__':

    print("Running tokenizer.py ...")

    tokenizer = get_gpt2_tokenizer()

    sentence = "Is 1029310928407 a multiple of 3?"
    print("      Sentence:", sentence)
    output = tokenizer.encode(sentence)
    print("After encoding:", output.tokens)
