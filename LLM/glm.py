'''
This script creates a CLI demo with transformers backend for the glm-4-9b model,
allowing users to interact with the model through a command-line interface.

Usage:
- Run the script to start the CLI demo.
- Interact with the model by typing questions and receiving responses.

Note: The script includes a modification to handle markdown to plain text conversion,
ensuring that the CLI interface displays formatted text correctly.

If you use flash attention, you should install the flash-attn and  add attn_implementation='flash_attention_2' in model loading.
'''

import os
import torch
from threading import Thread
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModel
import json
import re

## If use peft model.
# def load_model_and_tokenizer(model_dir, trust_remote_code: bool = True):
#     if (model_dir / 'adapter_config.json').exists():
#         model = AutoModel.from_pretrained(
#             model_dir, trust_remote_code=trust_remote_code, device_map='auto'
#         )
#         tokenizer_dir = model.peft_config['default'].base_model_name_or_path
#     else:
#         model = AutoModel.from_pretrained(
#             model_dir, trust_remote_code=trust_remote_code, device_map='auto'
#         )
#         tokenizer_dir = model_dir
#     tokenizer = AutoTokenizer.from_pretrained(
#         tokenizer_dir, trust_remote_code=trust_remote_code, use_fast=False
#     )
#     return model, tokenizer

initialPrompt = '''
你是一个心理学分析专家。接下来，我会提供一条微博热搜的标题、具体内容以及热搜下的一条评论。你的任务是根据这些信息对微博评论进行心理学分析，将问题分类，并根据严重程度进行标记。

分类包括十一个维度：

1: 器质性（包括症状性）精神障碍
2: 因使用精神活性物质导致的精神和行为障碍
3: 精神分裂症、分裂型和妄想症
4: 心境[情感]障碍
5: 神经性、应激性及躯体形式障碍
6: 与生理紊乱和身体因素相关的行为综合征
7: 成年人的人格和行为障碍
8: 智力迟钝
9: 心理发育障碍
10: 通常在儿童和青少年时期发病的行为和情绪障碍
11: 消极程度

**输入格式**：
json
{
  'title': '微博热搜标题',
  'comment': '热搜下的一条微博评论'
}


**输出格式**：
json
{
    ‘消极程度’： 0，
    '器质性（包括症状性）精神障碍': 0,
    '因使用精神活性物质导致的精神和行为障': 0,
    '精神分裂症、分裂型和妄想症': 0,
    '心境[情感]障碍': 0,
    '神经性、应激性及躯体形式障碍': 0,
    '与生理紊乱和身体因素相关的行为综合征': 0,
    '成年人的人格和行为障碍': 0,
    '智力迟钝': 0,
    '心理发育障碍': 0,
    '通常在儿童和青少年时期发病的行为和情绪障碍': 0,
}

请除了以上的格式其他的不要输出。


请根据输入信息，使用1-5的评分系统来表示各子类的可能性，其中1表示可能性最小，5表示可能性最大。如果某个子类不适用或没有明显迹象，请使用0表示。{
  'title': '微博热搜标题',
  'description': '微博热搜的具体内容',
  'comment': '热搜下的一条微博评论'
}


请根据输入信息，使用1-5的评分系统来表示各子类的可能性，其中1表示可能性最小，5表示可能性最大。如果某个子类不适用或没有明显迹象，请使用0表示。
'''

class StopOnTokens(StoppingCriteria):
    def __init__(self, model):
        self.model = model

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = self.model.config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class GLM():
    def __init__(self):
        # self.MODEL_PATH = os.environ.get('MODEL_PATH', '/home/ubuntu/LLaMA-Factory-main/output/glm_finetuned')
        self.MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4-9b-chat')

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_PATH,
            trust_remote_code=True,
            encode_special_tokens=True
        )

        self.model = AutoModel.from_pretrained(
            self.MODEL_PATH,
            trust_remote_code=True,
            # attn_implementation='flash_attention_2', # Use Flash Attention
            # torch_dtype=torch.bfloat16, #using flash-attn must use bfloat16 or float16
            device_map='auto'
        ).eval()
         

        self.history = []
        self.max_length = 8192
        self.top_p = 0.8
        self.temperature = 0.6
        self.stop = StopOnTokens(self.model)
        self.timer = 0

        self.run(initialPrompt, True)

    def reset(self):
        self.history = []
        self.run(initialPrompt, True)

    def run(self, input_words, skip = False):
        if self.timer > 8:
            self.timer = 0
            self.reset()
        self.history.append([input_words, ''])

        messages = []
        for idx, (user_msg, model_msg) in enumerate(self.history):
            if idx == len(self.history) - 1 and not model_msg:
                messages.append({'role': 'user', 'content': user_msg})
                break
            if user_msg:
                messages.append({'role': 'user', 'content': user_msg})
            if model_msg:
                messages.append({'role': 'assistant', 'content': model_msg})

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors='pt'
        ).to(self.model.device)


        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=True
        )
        

        generate_kwargs = {
            'input_ids': model_inputs,
            'streamer': streamer,
            'max_new_tokens': self.max_length,
            'do_sample': True,
            'top_p': self.top_p,
            'temperature': self.temperature,
            'stopping_criteria': StoppingCriteriaList([self.stop]),
            'repetition_penalty': 1.2,
            'eos_token_id': self.model.config.eos_token_id,
        }
        

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        output = ''
        for new_token in streamer:
            if new_token:
                if not skip:
                    output += new_token
                self.history[-1][1] += new_token

        self.history[-1][1] = self.history[-1][1].strip()
        # print(output)
        if not skip:
            self.timer += 1
            try:
                dictionary = json.loads(output[output.find("{"):output.find("}") + 1])
                return dictionary
            except:
                print("something went wrong\noutput:\n" + output)
                self.run(input_words)
        return
                
