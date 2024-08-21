from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-code-instruct-3b")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
# tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b", token="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")
print(tokenizer)
print(tokenizer.bos_token_id,
            tokenizer.encode("<｜fim▁hole｜>", add_special_tokens=False),
            tokenizer.encode("<｜fim▁begin｜>", add_special_tokens=False),
            tokenizer.encode("<｜fim▁end｜>", add_special_tokens=False),
            tokenizer.encode("<pad>", add_special_tokens=False)
        )
# print(tokenizer.encode("<fim_suffix>"),
#             tokenizer.encode("<fim_prefix>"),
#             tokenizer.encode("<fim_middle>"),
#             tokenizer.encode("<fim_pad>"))
