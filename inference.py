from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# load the original model first
MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
OUTPUT_DIR = "fim_llama"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=None,
    device_map=None,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda()

# merge fine-tuned weights with the base model
# peft_model_id = f"ardalaaan/{OUTPUT_DIR}"
# model = PeftModel.from_pretrained(base_model, peft_model_id)
# model.merge_and_unload()


def get_code_completion(prefix, suffix):
    text = f"""<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"""
    model.eval()
    outputs = model.generate(
        input_ids=tokenizer(text, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=128,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.0,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def get_base_code_completion(prefix, suffix):
    text = f"""<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"""
    MODEL.eval()
    outputs = MODEL.generate(
        input_ids=tokenizer(text, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=128,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.0,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


prefix = """from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
peft_config = LoraConfig(
"""
suffix = """"""

print(get_base_code_completion(prefix, suffix))

