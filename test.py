!pip install peft torch transformers datasets huggingface_hub
###########################################################################
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)

from peft import PeftModel
import torch

# load the original model first
MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=None,
    device_map=None,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda()

# merge fine-tuned weights with the base model
peft_model_id = f"ardalaaan/fim_llama"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.merge_and_unload()
###########################################################################
from huggingface_hub import notebook_login
notebook_login()

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
#########################################################################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataclasses import dataclass, field
from typing import Optional
import contextlib

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

model = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=None,
    device_map=None,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# model = model.merge_and_unload()
if not hasattr(model, "hf_device_map"):
    model.cuda()

base_model = model

model_id = "ardalaaan/fim_llama"
model = PeftModel.from_pretrained(model, model_id)
model.add_weighted_adapter(["personal_copilot"], [0.8], "best_personal_copilot")
model.set_adapter("best_personal_copilot")


def get_code_completion(prefix, suffix):
    text = prompt = f"""<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"""
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
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]


def get_base_code_completion(prefix, suffix):
    text = prompt = f"""<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"""
    base_model.eval()
    outputs = base_model.generate(
        input_ids=tokenizer(text, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=128,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.0,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
#############################################################################################

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataclasses import dataclass, field
from typing import Optional
import contextlib

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

model = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=None,
    device_map=None,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# model = model.merge_and_unload()
if not hasattr(model, "hf_device_map"):
    model.cuda()

# model_id = "ardalaaan/fim_llama"
# model = PeftModel.from_pretrained(model, model_id, adapter_name="personal_copilot")
# model.add_weighted_adapter(["personal_copilot"], [0.8], "best_personal_copilot")
# model.set_adapter("best_personal_copilot")


def get_code_completion(prefix, suffix):
    text = prompt = f"""<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"""
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
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]


def get_base_code_completion(prefix, suffix):
    text = prompt = f"""<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"""
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
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
#################################################################################

prefix = """from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, training_dataloader, scheduler = """

suffix = """"""
print(get_base_code_completion(prefix, suffix))






##################################################################################
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[5], line 17
     15 # merge fine-tuned weights with the base model
     16 peft_model_id = f"ardalaaan/fim_llama"
---> 17 model = PeftModel.from_pretrained(base_model, peft_model_id)
     18 model.merge_and_unload()

File /usr/local/lib/python3.10/dist-packages/peft/peft_model.py:545, in PeftModel.from_pretrained(cls, model, model_id, adapter_name, is_trainable, config, autocast_adapter_dtype, ephemeral_gpu_offload, **kwargs)
    540 else:
    541     model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](
    542         model, config, adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
    543     )
--> 545 model.load_adapter(
    546     model_id, adapter_name, is_trainable=is_trainable, autocast_adapter_dtype=autocast_adapter_dtype, **kwargs
    547 )
    549 return model

File /usr/local/lib/python3.10/dist-packages/peft/peft_model.py:1117, in PeftModel.load_adapter(self, model_id, adapter_name, is_trainable, torch_device, autocast_adapter_dtype, ephemeral_gpu_offload, **kwargs)
   1115 # load the weights into the model
   1116 ignore_mismatched_sizes = kwargs.get("ignore_mismatched_sizes", False)
-> 1117 load_result = set_peft_model_state_dict(
   1118     self, adapters_weights, adapter_name=adapter_name, ignore_mismatched_sizes=ignore_mismatched_sizes
   1119 )
   1120 if (
   1121     (getattr(self, "hf_device_map", None) is not None)
   1122     and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
   1123     and len(self.peft_config) == 1
   1124 ):
   1125     device_map = kwargs.get("device_map", "auto")

File /usr/local/lib/python3.10/dist-packages/peft/utils/save_and_load.py:395, in set_peft_model_state_dict(model, peft_model_state_dict, adapter_name, ignore_mismatched_sizes)
    390     raise NotImplementedError
    392 peft_model_state_dict, mismatched_keys = _find_mismatched_keys(
    393     model, peft_model_state_dict, ignore_mismatched_sizes=ignore_mismatched_sizes
    394 )
--> 395 load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    396 if config.is_prompt_learning:
    397     model.prompt_encoder[adapter_name].embedding.load_state_dict(
    398         {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
    399     )

File /usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py:2152, in Module.load_state_dict(self, state_dict, strict, assign)
   2147         error_msgs.insert(
   2148             0, 'Missing key(s) in state_dict: {}. '.format(
   2149                 ', '.join(f'"{k}"' for k in missing_keys)))
   2151 if len(error_msgs) > 0:
-> 2152     raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
   2153                        self.__class__.__name__, "\n\t".join(error_msgs)))
   2154 return _IncompatibleKeys(missing_keys, unexpected_keys)

RuntimeError: Error(s) in loading state_dict for PeftModelForCausalLM:
	size mismatch for base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.0.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.0.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.0.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.0.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.0.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.1.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.1.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.1.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.1.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.1.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.1.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.1.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.1.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.1.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.1.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.1.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.1.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.1.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.1.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.2.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.2.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.2.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.2.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.2.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.2.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.2.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.2.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.2.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.2.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.2.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.2.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.2.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.2.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.3.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.3.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.3.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.3.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.3.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.3.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.3.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.3.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.3.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.3.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.3.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.3.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.3.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.3.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.4.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.4.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.4.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.4.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.4.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.4.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.4.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.4.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.4.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.4.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.4.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.4.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.4.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.4.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.5.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.5.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.5.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.5.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.5.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.5.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.5.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.5.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.5.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.5.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.5.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.5.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.5.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.5.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.6.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.6.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.6.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.6.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.6.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.6.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.6.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.6.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.6.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.6.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.6.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.6.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.6.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.6.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.7.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.7.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.7.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.7.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.7.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.7.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.7.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.7.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.7.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.7.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.7.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.7.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.7.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.7.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.8.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.8.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.8.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.8.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.8.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.8.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.8.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.8.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.8.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.8.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.8.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.8.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.8.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.8.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.9.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.9.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.9.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.9.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.9.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.9.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.9.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.9.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.9.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.9.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.9.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.9.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.9.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.9.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.10.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.10.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.10.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.10.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.10.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.10.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.10.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.10.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.10.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.10.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.10.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.10.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.10.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.10.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.11.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.11.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.11.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.11.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.11.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.11.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.11.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.11.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.11.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.11.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.11.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.11.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.11.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.11.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.12.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.12.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.12.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.12.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.12.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.12.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.12.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.12.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.12.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.12.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.12.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.12.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.12.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.12.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.13.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.13.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.13.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.13.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.13.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.13.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.13.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.13.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.13.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.13.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.13.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.13.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.13.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.13.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.14.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.14.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.14.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.14.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.14.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.14.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.14.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.14.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.14.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.14.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.14.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.14.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.14.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.14.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.15.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.15.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.15.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.15.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.15.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.15.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.15.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.15.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.15.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.15.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.15.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.15.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.15.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.15.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.16.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.16.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.16.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.16.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.16.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.16.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.16.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.16.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.16.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.16.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.16.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.16.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.16.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.16.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.17.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.17.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.17.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.17.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.17.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.17.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.17.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.17.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.17.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.17.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.17.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.17.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.17.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.17.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.18.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.18.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.18.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.18.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.18.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.18.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.18.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.18.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.18.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.18.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.18.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.18.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.18.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.18.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.19.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.19.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.19.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.19.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.19.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.19.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.19.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.19.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.19.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.19.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.19.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.19.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.19.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.19.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.20.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.20.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.20.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.20.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.20.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.20.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.20.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.20.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.20.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.20.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.20.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.20.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.20.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.20.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.21.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.21.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.21.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.21.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.21.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.21.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.21.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.21.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.21.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.21.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.21.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.21.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.21.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.21.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.22.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.22.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.22.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.22.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.22.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.22.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.22.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.22.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.22.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.22.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.22.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.22.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.22.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.22.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.23.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.23.self_attn.q_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.23.self_attn.k_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.23.self_attn.k_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.23.self_attn.v_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.23.self_attn.v_proj.lora_B.default.weight: copying a param with shape torch.Size([1024, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.23.self_attn.o_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.23.self_attn.o_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).
	size mismatch for base_model.model.model.layers.23.mlp.gate_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.23.mlp.gate_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.23.mlp.up_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 4096]) from checkpoint, the shape in current model is torch.Size([16, 2048]).
	size mismatch for base_model.model.model.layers.23.mlp.up_proj.lora_B.default.weight: copying a param with shape torch.Size([14336, 16]) from checkpoint, the shape in current model is torch.Size([5504, 16]).
	size mismatch for base_model.model.model.layers.23.mlp.down_proj.lora_A.default.weight: copying a param with shape torch.Size([16, 14336]) from checkpoint, the shape in current model is torch.Size([16, 5504]).
	size mismatch for base_model.model.model.layers.23.mlp.down_proj.lora_B.default.weight: copying a param with shape torch.Size([4096, 16]) from checkpoint, the shape in current model is torch.Size([2048, 16]).