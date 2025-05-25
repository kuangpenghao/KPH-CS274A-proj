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

    # Step 1: 清理词汇表
    # 规则:
    # 1. 删除作为“纯数字”且长度大于1的词元。
    # 2. 删除形式为 "连续任意位数非数字" + "连续大于1位的纯数字" 的词元。
    def should_remove_from_vocab(token_str: str) -> bool:
        if not token_str:
            return False

        # 规则 1: 纯数字且长度大于1
        if token_str.isdigit():
            return len(token_str) > 1

        # 规则 2: "连续任意位数非数字" + "连续大于1位的纯数字"
        # Find the start of the trailing pure digit segment
        idx = len(token_str)
        while idx > 0 and token_str[idx-1].isdigit():
            idx -= 1
        
        prefix = token_str[0:idx]
        suffix = token_str[idx:]

        if not suffix: # No trailing digits, or token is all non-digits
            return False

        # suffix is guaranteed to be all digits here, and non-empty.
        # prefix might be empty if original token_str was all digits (handled by rule 1).
        
        if len(suffix) > 1: # "连续大于1位的纯数字"
            if bool(prefix): # Prefix must be non-empty
                # Check if prefix is "连续任意位数非数字"
                is_prefix_all_nondigits = all(not c.isdigit() for c in prefix)
                if is_prefix_all_nondigits:
                    return True
        return False

    current_vocab_keys = list(vocab.keys()) # 保持顺序
    keys_to_keep_in_vocab = []
    for token_k in current_vocab_keys:
        if not should_remove_from_vocab(token_k):
            keys_to_keep_in_vocab.append(token_k)

    final_vocab = {}
    new_idx = 0
    for token_k in keys_to_keep_in_vocab:
        final_vocab[token_k] = new_idx
        new_idx += 1
    
    vocab.clear()
    vocab.update(final_vocab)

    # Step 2: 清理合并规则
    # 规则:
    # 1. 删除 "纯数字" + "纯数字" 的合并。
    # 2. 删除 ("连续任意位数非数字"+"纯数字") + "纯数字" 的合并。
    new_cleaned_merges = []
    for p1, p2 in merges:
        merged_token = p1 + p2

        if not (p1 in vocab and p2 in vocab and merged_token in vocab):
            continue

        def is_pure_digit(token: str) -> bool:
            return bool(token) and token.isdigit()

        def is_prefix_nondigit_suffix_puredigit(token: str) -> bool:
            if not token or token.isdigit(): # Must not be empty or pure digit
                return False
            
            idx = len(token)
            while idx > 0 and token[idx-1].isdigit():
                idx -= 1
            
            prefix = token[0:idx]
            suffix = token[idx:]

            if not prefix or not suffix: # Must have both parts
                return False
            
            return all(not c.isdigit() for c in prefix) # Prefix all non-digits, suffix is all digits (guaranteed by loop)

        p1_is_pure = is_pure_digit(p1)
        p2_is_pure = is_pure_digit(p2)
        p1_is_nondigit_plus_digit = is_prefix_nondigit_suffix_puredigit(p1)

        delete_this_merge = False

        # 规则 1: "纯数字" + "纯数字"
        if p1_is_pure and p2_is_pure:
            delete_this_merge = True
        
        # 规则 2: ("连续任意位数非数字"+"纯数字") + "纯数字"
        if not delete_this_merge:
            if p1_is_nondigit_plus_digit and p2_is_pure:
                delete_this_merge = True
        
        if not delete_this_merge:
            new_cleaned_merges.append((p1, p2))

    merges[:] = new_cleaned_merges


if __name__ == '__main__':

    print("Running tokenizer.py ...")

    tokenizer = get_gpt2_tokenizer()

    sentence = "Is 1029310928407 a multiple of 3?"
    print("      Sentence:", sentence)
    output = tokenizer.encode(sentence)
    print("After encoding:", output.tokens)
