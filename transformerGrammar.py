# transformerGrammar.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)
# The question was created by Haoyu Du (duhy@shanghaitech.edu.cn).


import util

import torch
import torch.nn.functional as F

from datasets import load_dataset, Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments, PreTrainedModel
from transformers.models.gpt_neo import GPTNeoConfig, GPTNeoForCausalLM


class InvalidTreeError(Exception):
    pass


def mapping_function(example: dict) -> dict:
    actions = example["actions"]
    inputs = []
    labels = []
    position_ids = []
    
    # 1. 验证树的合法性
    #检查括号是否匹配，以及非终结符和终结符的规则
    
    if not actions or len(actions) < 2:
        raise InvalidTreeError("Action sequence is too short.")
    if actions[0] != "<s>" or actions[-1] != "</s>":
        raise InvalidTreeError("Action sequence must start with <s> and end with </s>.")

    open_brackets_count = 0
    nt_name_stack = [] # Stores just the name, e.g., "S"
    
    # Iterate tokens between <s> and </s>
    for i in range(1, len(actions) - 1):
        token = actions[i]
        
        # Basic token validation
        if not isinstance(token, str) or not token:
            raise InvalidTreeError(f"Invalid or empty token found in actions sequence: {token}")

        is_ont = token.startswith("(")
        is_cnt = token.endswith(")")

        if is_ont and is_cnt: # e.g. "()" or "(A)" which are not typical for this problem's format
             raise InvalidTreeError(f"Token '{token}' looks like both an opening and closing non-terminal.")
        
        if is_ont:
            nt_name = token[1:]
            if not nt_name: # Check for "()"
                raise InvalidTreeError(f"Opening non-terminal '{token}' has an empty name.")
            open_brackets_count += 1
            nt_name_stack.append(nt_name)
            
            # Check for immediate closure: (X X)
            if i + 1 < len(actions) - 1: # Ensure next token is also not </s>
                next_token = actions[i+1]
                if next_token.endswith(")") and not next_token.startswith("("): # next_token is a CNT
                    if next_token[:-1] == nt_name:
                        raise InvalidTreeError(f"Non-terminal {token} is immediately closed by {next_token}, indicating an empty production.")
        elif is_cnt:
            nt_name = token[:-1]
            if not nt_name: # Check for "X)" where X is empty
                 raise InvalidTreeError(f"Closing non-terminal '{token}' has an empty name part.")
            if open_brackets_count == 0 or not nt_name_stack or nt_name_stack[-1] != nt_name:
                expected_nt = nt_name_stack[-1] if nt_name_stack else "None (no open non-terminal)"
                raise InvalidTreeError(f"Mismatched or unexpected closing non-terminal '{token}'. Expected to close '{expected_nt}'.")
            nt_name_stack.pop()
            open_brackets_count -= 1
        else: # Terminal
            if open_brackets_count == 0: # Terminal outside any open non-terminal
                raise InvalidTreeError(f"Terminal '{token}' appears outside of any non-terminal structure.")
            if not token.strip(): # Check for empty or whitespace-only terminals
                 raise InvalidTreeError("Empty or whitespace-only terminal token found.")


    if open_brackets_count != 0:
        raise InvalidTreeError("Unbalanced non-terminals at the end of the sequence (more opening than closing).")
    
    # The original checks like:
    # if len(actions) > 2 and actions[-2].startswith("("): raise InvalidTreeError(...)
    # if len(actions) > 2 and actions[1].endswith(")"): raise InvalidTreeError(...)
    # are implicitly covered by the loop logic above. 
    # For example, if actions[-2] is '(', open_brackets_count will not be 0 at the end.
    # If actions[1] is ')', open_brackets_count will be 0, nt_name_stack empty, leading to error on ')' processing.

    # 2. 处理输入 (inputs) 和 3. 处理输出 (labels)
    for token in actions:
        inputs.append(token)
        labels.append(token)
        if token.endswith(")") and token != "</s>": # 排除 </s>
            inputs.append(token) # 复制闭合非终结符
            labels.append("<pad>") # 插入 <pad>

    # 4. 计算绝对位置 (position_ids)
    # <s> 和 </s> 的深度为 0
    # 开非终结符 (X 深度加1，闭非终结符 X) 深度减1
    # 终结符的深度与最近的开非终结符相同
    # 复制的闭合非终结符的深度与原闭合非终结符相同
    
    processed_inputs_for_pos = []
    temp_actions = [] # 用于计算position_ids的临时actions，不包含复制的闭合非终结符
    for token in actions:
        temp_actions.append(token)
        if token.endswith(")") and token != "</s>":
            pass # 在计算原始位置时，不考虑复制的闭合非终结符和<pad>
            
    current_depth = 0
    original_pos_ids = []
    for token in temp_actions:
        if token == "<s>" or token == "</s>":
            original_pos_ids.append(0)
        elif token.startswith("("):
            original_pos_ids.append(current_depth)
            current_depth += 1
        elif token.endswith(")"):
            current_depth -= 1
            original_pos_ids.append(current_depth)
        else: # Terminals
            original_pos_ids.append(current_depth)
            
    # 现在根据 inputs 列表构建最终的 position_ids
    # 遍历原始 actions，当遇到闭合非终结符时，其在 inputs 中对应两个元素
    # labels 中的 <pad> 不需要 position_id，但这里是针对 inputs 的
    
    actions_ptr = 0
    for token_in_inputs in inputs:
        original_token = actions[actions_ptr]
        position_ids.append(original_pos_ids[actions_ptr])
        if token_in_inputs == original_token: # 当前token是原始token
            if not (token_in_inputs.endswith(")") and token_in_inputs != "</s>" and actions_ptr + 1 < len(actions) and inputs[inputs.index(token_in_inputs, actions_ptr) +1] == token_in_inputs ):
                 # 如果不是一个被复制的闭合非终结符的原始token，或者它是最后一个token，则指针前进
                 # 这个条件有点复杂，主要是为了处理 inputs 中重复的闭合非终结符
                 # 简单来说，如果当前 inputs token 和 actions token 相同，并且它不是一个后面跟着复制品的闭合非终结符，那么 actions_ptr 前进
                 # 或者更简单：如果当前 inputs token 是 actions token，并且它不是一个后面会紧跟着一个相同 token 的闭合非终结符
                
                # 重新思考这里的逻辑：
                # 我们需要将 original_pos_ids 的值映射到 inputs 列表上
                # inputs 的长度可能大于 actions
                # 当 inputs[i] 是 actions[j] 的复制时，pos_ids[i] = original_pos_ids[j]
                
                # 简化逻辑：
                # position_ids 的长度应该和 inputs 一样
                # 我们已经有了 original_pos_ids 对应 actions
                # 当 inputs[i] 是 actions[j] 时， position_ids[i] = original_pos_ids[j]
                # 当 inputs[i] 是 actions[j] 的复制品（即闭合非终结符的第二个实例）时，position_ids[i] = original_pos_ids[j]
                
                # 我们重新构建 position_ids
                pass # position_ids 已经在下面正确构建了

        if token_in_inputs == actions[actions_ptr] and not (token_in_inputs.endswith(")") and token_in_inputs != "</s>" and inputs.count(token_in_inputs) > actions.count(token_in_inputs) and inputs[inputs.index(token_in_inputs) + 1 if inputs.index(token_in_inputs) +1 < len(inputs) else inputs.index(token_in_inputs)] == token_in_inputs) :
             actions_ptr +=1
             if actions_ptr >= len(actions): # 防止越界
                 actions_ptr = len(actions) -1


    # 重新计算 position_ids 以确保正确性
    position_ids = []
    current_depth = 0
    # 栈，用于跟踪非终结符的起始位置，以便在闭合时正确设置深度
    # (token_index_in_inputs, depth_at_opening)
    # 这个栈的目的是为了处理嵌套结构，确保闭合标签的深度正确
    
    # 简化版：直接根据 inputs 和原始的深度逻辑计算
    # 假设 inputs 已经是 ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', 'NP)', '(VP', 'sings', 'VP)', 'VP)', 'S)', 'S)', '</s>']
    # 原始 actions: ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', '(VP', 'sings', 'VP)', 'S)', '</s>']
    # 原始 pos:    [0,   0,    1,     2,     2,      2,      1,     1,       2,      1,    0,     0]
    
    # 映射规则：
    # inputs 中的 token 如果是原始 actions 中的 token，则取其原始 pos
    # inputs 中的 token 如果是复制的闭合非终结符，则取其对应的原始闭合非终结符的 pos
    
    temp_pos_map = {} # 记录原始 action token 在 actions 中的索引及其 pos
    _current_depth_for_orig = 0
    _original_actions_pos = []
    for token in actions:
        if token == "<s>" or token == "</s>":
            _original_actions_pos.append(0)
        elif token.startswith("("):
            _original_actions_pos.append(_current_depth_for_orig)
            _current_depth_for_orig += 1
        elif token.endswith(")"):
            _current_depth_for_orig -= 1
            _original_actions_pos.append(_current_depth_for_orig)
        else: # Terminals
            _original_actions_pos.append(_current_depth_for_orig)

    action_idx_map = {} # token_str -> list of original indices in `actions`
    for i, token in enumerate(actions):
        if token not in action_idx_map:
            action_idx_map[token] = []
        action_idx_map[token].append(i)

    # 为 inputs 中的每个 token 找到其对应的原始 action token 的位置，然后取其深度
    # 对于重复的闭合非终结符，它们共享相同的原始 token 和深度
    
    # Example:
    # actions: ["<s>", "(A", "a", "A)", "</s>"] -> pos: [0,0,1,0,0]
    # inputs:  ["<s>", "(A", "a", "A)", "A)", "</s>"]
    # labels:  ["<s>", "(A", "a", "A)", "<pad>", "</s>"]
    # pos_ids: ["<s>":0, "(A":0, "a":1, "A)":0, "A)":0, "</s>":0]

    current_actions_ptr = 0
    for i, token_in_inputs in enumerate(inputs):
        # 找到这个 token_in_inputs 对应于 actions 中的哪个 token
        # 如果 token_in_inputs 是一个闭合非终结符，并且它是被复制的那个
        # 那么它对应于 actions 中那個原始的闭合非终结符
        
        # 逻辑：如果 inputs[i] == actions[current_actions_ptr]，则 pos_ids.append(_original_actions_pos[current_actions_ptr])
        # 如果 inputs[i] 是闭合非终结符且 inputs[i] == inputs[i-1]，则它是复制的，pos_ids.append(pos_ids[i-1])
        # 否则，current_actions_ptr++ （在非复制情况下）
        
        if i > 0 and token_in_inputs.endswith(")") and token_in_inputs != "</s>" and token_in_inputs == inputs[i-1]:
            # 这是被复制的闭合非终结符
            position_ids.append(position_ids[i-1])
        else:
            position_ids.append(_original_actions_pos[current_actions_ptr])
            current_actions_ptr +=1
            
    # 5. 生成注意力掩码 (attention_mask)
    # STACK/COMPOSE attention.
    # 掩码形状 (len(inputs), len(inputs))
    # <s> 和 </s> 的掩码全0 (题目提示：</s> 的掩码全0，但通常 <s> 也会特殊处理或有自己的规则)
    # 我们假设 <s> 的掩码也全0，或者只关注自身
    # 对于其他 token i，它可以 attend to token j if:
    #   - j <= i (causal attention)
    #   - 并且满足 STACK/COMPOSE 规则
    # STACK: 一个 token 可以 attend to 所有在它之前且在同一层级或更高层级的未闭合的非终结符的开括号，以及这些非终结符内部的所有内容。
    # COMPOSE: 当一个闭合非终结符 Y) 出现时，它和它对应的开非终结符 (Y 构成一个单元。
    #          后续的 token 可以 attend to 这个 (Y ... Y) 单元的代表，通常是 (Y。
    #          或者，对于 Y) 自身，它可以 attend to (Y 以及 (Y 内部的所有内容。
    #
    # 简化实现基于论文中的描述和常见做法：
    # - 每个 token attend to itself.
    # - 每个 token attend to <s>.
    # - 对于 (X (Y ... Y) Z ... X) 结构:
    #   - X) can attend to (X and everything inside (X ... X)
    #   - Tokens inside (Y ... Y) can attend to (Y and everything inside (Y ... Y) up to themselves (causal)
    #   - Z can attend to (X and the composed (Y...Y) (represented by (Y) )
    #
    # 这是一个复杂的逻辑，通常需要维护一个栈来跟踪当前的非终结符。
    # 鉴于这是一个编程题，我们先实现一个基础的因果掩码，然后尝试加入 STACK/COMPOSE 的简化规则。
    # 如果完全按照论文，会非常复杂。
    #
    # 示例中的掩码是关键。
    # 'inputs': ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', 'NP)', '(VP', 'sings', 'VP)', 'VP)', 'S)', 'S)', '</s>']
    # 长度为 15
    
    # 规则：
    # 1. 对角线为1 (token attend to itself)
    # 2. <s> 和 </s> 的行全为0 (除了 <s> attend to <s>, </s> attend to </s> 如果遵循规则1)
    #    题目明确指出 </s> 的 attention_mask 全为0。对于 <s>，通常它只 attend to itself 或者不 attend to anything else in a causal LM.
    #    我们先让 <s> 只 attend to itself.
    # 3. 对于 token i (不是 <s> 或 </s>):
    #    它可以 attend to token j (j < i) if:
    #    - token j 是 <s>
    #    - token j 是一个开放的非终结符 (X 且 X 尚未关闭 (在 j 和 i 之间没有对应的 X) )
    #    - token j 是一个终结符，且这个终结符与 token i 在同一个直接的非终结符父节点下。
    #    - 如果 token i 是一个闭合非终结符 X)，它可以 attend to 对应的 (X 以及 (X 内部的所有 token。
    #    - 如果 token i 是一个复制的闭合非终结符 X)，它的行为和原始的 X) 类似。

    # 我们将使用一个更直接的方法，基于栈来构建掩码，模拟论文中的 Figure 2 行为。
    # The attention mask of '</s>' is all 0s.
    
    n = len(inputs)
    attention_mask = torch.zeros((n, n), dtype=torch.long)

    for i in range(n): # query token index
        if inputs[i] == "</s>":
            continue # Rule: attention_mask of '</s>' is all 0s.

        # Rule 1: Token attends to itself (except </s>, handled above)
        attention_mask[i, i] = 1

        # Rule 2: All tokens (except </s>) attend to <s>
        if i > 0 and inputs[0] == "<s>":
            attention_mask[i, 0] = 1

        # STACK/COMPOSE logic for j < i
        # For query token inputs[i], iterate through potential key tokens inputs[j]
        
        # Store indices of open non-terminals encountered *before* current query token i
        # This stack helps identify ancestors and completed constituents.
        # (ont_index_in_inputs, nt_name)
        active_ont_stack = [] 
        
        for j in range(i): # key token index, j < i
            key_token = inputs[j]

            if key_token.startswith("(") and not key_token.endswith(")"): # Key is an ONT
                active_ont_stack.append((j, key_token[1:]))
                # Rule: Attend to open non-terminals (ancestors)
                attention_mask[i, j] = 1
            elif key_token.endswith(")") and not key_token.startswith("(") and key_token != "</s>": # Key is a CNT
                # This CNT `key_token` at index `j` closes a constituent.
                # Query `i` can attend to the opening of this constituent (COMPOSE rule).
                if active_ont_stack and active_ont_stack[-1][1] == key_token[:-1]:
                    # The ONT that `key_token` closes is active_ont_stack[-1][0]
                    ont_idx_of_closed_constituent = active_ont_stack[-1][0]
                    attention_mask[i, ont_idx_of_closed_constituent] = 1
                    active_ont_stack.pop() # Pop it as it's now closed *before* query i
                # else: Mismatched CNT, should have been caught by validation. Or stack is empty.

        # Rule: If query inputs[i] is a CNT X), it attends to its matching (X and everything in between.
        # This rule is dominant and applies to j <= i for the constituent.
        if inputs[i].endswith(")") and not inputs[i].startswith("(") and inputs[i] != "</s>":
            current_cnt_name = inputs[i][:-1]
            # Find the matching ONT for inputs[i]
            # Scan backwards from i-1 to find the corresponding (X
            # This needs to correctly identify the *specific* opening bracket for *this* closing bracket,
            # handling nested structures like (A (B B) A).
            
            # To find the matching ONT for inputs[i] (a CNT):
            # We scan from the beginning up to i, maintaining a stack of ONTs.
            # When inputs[i] is encountered, its matching ONT is the one that was pushed
            # when the corresponding (NT was seen and hasn't been popped by an intermediate CNT.
            
            temp_ont_matcher_stack = [] # (ont_index, ont_name)
            matching_ont_index_for_current_cnt = -1

            for k in range(i + 1): # Iterate up to and including the current CNT inputs[i]
                token_k = inputs[k]
                if token_k.startswith("(") and not token_k.endswith(")"):
                    temp_ont_matcher_stack.append((k, token_k[1:]))
                elif token_k.endswith(")") and not token_k.startswith("(") and token_k != "</s>":
                    # This is a CNT at index k
                    if temp_ont_matcher_stack and temp_ont_matcher_stack[-1][1] == token_k[:-1]:
                        # This CNT (token_k) matches the ONT at the top of the stack
                        popped_ont_index = temp_ont_matcher_stack.pop()[0]
                        if k == i: # This is the current query CNT inputs[i]
                            matching_ont_index_for_current_cnt = popped_ont_index
                            break # Found the ONT for our query CNT
            
            if matching_ont_index_for_current_cnt != -1:
                for k_constituent in range(matching_ont_index_for_current_cnt, i + 1):
                    attention_mask[i, k_constituent] = 1

    # Ensure <s> only attends to itself if no other rules apply to its row (row 0).
    # Based on example, <s> seems to only attend to itself.
    # The general rule "token attends to self" already covers attention_mask[0,0]=1.
    # If other tokens should not attend to <s> unless specified, that's handled by default zero init.
    # The rule "all tokens attend to <s>" (attention_mask[i,0]=1 for i>0) is applied above.
    # Let's verify <s> row based on example if needed. The example mask shows <s> only attends to itself.
    # So, for row 0 (query is <s>), only attention_mask[0,0] should be 1.
    if n > 0 and inputs[0] == "<s>":
        for j_key in range(1, n): # Set all other entries in <s> row to 0
            attention_mask[0, j_key] = 0
        attention_mask[0,0] = 1 # Explicitly ensure <s> attends to itself only

    return {
        "inputs": inputs,
        "labels": labels,
        "position_ids": position_ids,
        "attention_mask": attention_mask
    }


def get_trainer(
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    train_dataset: Dataset
) -> Trainer:
    """
    Question:
        Create a Trainer object for the model. The Trainer is used to train the model on the dataset.
        Select the appropriate training arguments for the Trainer. For example, setting the proper learning rate,
        batch size, optimizer, learning rate scheduler, number of epochs, etc. would be a good idea.

    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use for the model.
        model (PreTrainedModel): The model to train.
        train_dataset (Dataset): The dataset to train on.

    Returns:
        trainer (Trainer): The Trainer object for the model.

    Example:
        >>> trainer = get_trainer(tokenizer, model, train_dataset)
        >>> trainer.train()
        >>> trainer.evaluate(train_dataset)
        {'eval_loss': 2.1234, ...}
    """

    def data_collator(features):
        """
        Data collator is to aggregate the features into a batch. You'll find it helpful when creating the Trainer.
        We simply pad the sequences but deal with attention mask seperately.
        """
        max_length = max([len(f["input_ids"]) for f in features])
        batch = {
            "input_ids": [],
            "labels": [],
            "position_ids": [],
            "attention_mask": [],
        }
        for f in features:
            input_ids = f["input_ids"]
            labels = f["labels"]
            position_ids = f["position_ids"]
            attention_mask = f["attention_mask"]
            seq_len = len(input_ids)

            input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
            labels += [-100] * (max_length - len(labels))
            position_ids += [0] * (max_length - len(position_ids))
            attention_mask = F.pad(torch.tensor(attention_mask), [0, max_length - seq_len, 0, max_length - seq_len])

            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
            batch["position_ids"].append(position_ids)
            batch["attention_mask"].append(attention_mask)

        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        batch["position_ids"] = torch.tensor(batch["position_ids"], dtype=torch.long)
        batch["attention_mask"] = torch.stack(batch["attention_mask"])

        return batch

    """YOUR CODE HERE"""
    util.raiseNotDefined()


def main():
    """This function trains a Transformer Grammar model based on GPT2 for the task of generative transition-based parsing."""
 
    ## Load the dataset from disk
    dataset = load_dataset("text", data_files="data/corpus.cc", split="train")


    ## Build the word tokenizer
    # Initialize tokenizer with special tokens
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))

    # Use the whitespace pre-tokenizer to split on whitespace
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # Build the vocabulary using WordLevelTrainer
    trainer = WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>"])
    tokenizer.train_from_iterator(dataset["text"], trainer=trainer)

    # Set the post-processor to add special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )

    # Convert to PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>'})


    ## Preprocess the dataset
    def tokenize_function(example):
        tokenized = tokenizer.tokenize(example["text"], add_special_tokens=True)
        return {"actions": tokenized}

    def convert_function(examples):
        input_ids = tokenizer(examples["inputs"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        labels = tokenizer(examples["labels"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        labels = [[(idx if idx != tokenizer.pad_token_id else -100) for idx in sent] for sent in labels]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": examples["position_ids"],
            "attention_mask": [[mask] for mask in examples["attention_mask"]],
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=["text"], load_from_cache_file=False)
    mapped_dataset = tokenized_dataset.map(mapping_function, batched=False, remove_columns=["actions"], load_from_cache_file=False)
    converted_dataset = mapped_dataset.map(convert_function, batched=True, remove_columns=["inputs"], load_from_cache_file=False)


    # Load the model
    # TODO: use GPT2 instead of GPTNeo when transformers 4.52.0 is released
    # We use GPTNeo here since the implementation of GPT2 has a bug and the fix has not been released yet.
    # GPTNeo is similar to GPT2 except that it uses local attention. We have disabled local attention in the config.
    config = GPTNeoConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_layers=6,
        num_heads=8,
        attention_types=[[["global"], 6]],
        activation_function="relu",
    )
    model = GPTNeoForCausalLM(config)


    # Training
    trainer = get_trainer(tokenizer, model, converted_dataset)
    trainer.train()
    metrics = trainer.evaluate(converted_dataset)

    print(metrics)


if __name__ == "__main__":
    main()
