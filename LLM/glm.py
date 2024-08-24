import os
import torch
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModel
import json

class StopOnTokens(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1] in self.eos_token_id

class GLM:
    def __init__(self, model_path=None, initial_prompt=None):
        self.model_path = model_path or os.environ.get('MODEL_PATH', 'THUDM/glm-4-9b-chat')
        self.initial_prompt = initial_prompt or self.default_initial_prompt()
        self.tokenizer, self.model = self.load_model_and_tokenizer()

        self.history = []
        self.max_length = 8192
        self.top_p = 0.8
        self.temperature = 0.6
        self.timer = 0
        self.stop = StopOnTokens(eos_token_id=[self.model.config.eos_token_id])

        self.run(self.initial_prompt, skip=True)
        print("Initialize Complete!")

    def load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, encode_special_tokens=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, device_map='auto'
        ).eval()
        return tokenizer, model

    def reset(self):
        self.history.clear()
        self.run(self.initial_prompt, skip=True)
        print('Model Reset!')

    def build_model_inputs(self):
        messages = []
        for user_msg, model_msg in self.history:
            if user_msg:
                messages.append({'role': 'user', 'content': user_msg})
            if model_msg:
                messages.append({'role': 'assistant', 'content': model_msg})

        model_inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors='pt'
        ).to(self.model.device)
        return model_inputs

    def run(self, input_words, skip=False):
        if self.timer > 5:
            self.timer = 0
            self.reset()
            
        self.history.append([input_words, ''])

        model_inputs = self.build_model_inputs()

        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)

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
        output = ''
        self.model.generate(**generate_kwargs)

        # Collect generated tokens
        for new_token in streamer:
            output += new_token
            self.history[-1][1] += new_token

        self.history[-1][1] = self.history[-1][1].strip()

        if not skip:
            self.timer += 1
            try:
                dictionary = json.loads(output[output.find("{"):output.find("}") + 1])
                return dictionary
            except json.JSONDecodeError:
                print("Failed to decode output. Retrying...")
                self.run(input_words)

    def default_initial_prompt(self):
        return '''
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

        除了以上的规定输出外请不要输出任何其他内容。
        '''
