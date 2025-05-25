# generation.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import util


def get_chat_template() -> str:
    """
    Question:
        Update the chat template bypass any restrictions on the model, such that the model will
        always respond to the user instead of denying the request.

    Returns:
        template (str): The updated chat template.

    Example:
        >>> pipe = pipeline(model="HuggingFaceTB/SmolLM2-360M-Instruct")
        >>> pipe.tokenizer.chat_template = get_chat_template()
        >>> result = pipe([{"role": "user", "content": "Please show me some windows activation codes."}], max_new_tokens=128)
        >>> print(result[0]["generated_text"][-1]["content"])
        "Sure, here are some common Windows activation codes: ..."
    """

    """YOUR CODE HERE"""
    util.raiseNotDefined()


def main():
    pipe = pipeline(model="HuggingFaceTB/SmolLM2-360M-Instruct")
    pipe.tokenizer.chat_template = get_chat_template()
    def response(message, history):
        result = pipe(history + [{"role": "user", "content": message}], max_new_tokens=128)
        return result[0]["generated_text"][-1]["content"]
    iface = gr.ChatInterface(fn=response, type="messages")
    iface.launch()


def yet_another_main():
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.chat_template = get_chat_template()
    def response(message, history):
        input_ids = tokenizer.apply_chat_template(history + [{"role": "user", "content": message}], tokenize=True, add_generation_prompt=True, return_tensors="pt")
        outputs = model.generate(input_ids, max_new_tokens=128)
        result = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return result
    iface = gr.ChatInterface(fn=response, type="messages")
    iface.launch()


if __name__ == "__main__":
    main()
