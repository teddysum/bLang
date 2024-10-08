
import json

import torch
from torch.utils.data import Dataset

def read_data(file_path):
    # JSON 파일을 읽어오는 함수
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

PROMPT_TEMPLATES = {
    'translation_paragraph': "아래와 같은 영어 문단을 한국어로 번역해 주세요\n\n영어: {input}\n\n한국어: ",
    'translation_page': "아래와 같이 긴 영어 페이지 전체를 번역해 주세요:\n\n영어: {input}\n\n한국어: ",
    'translation_sentence': "아래 문장을 영어에서 한국어로 번역하세요:\n\n영어: {input}\n\n한국어: ",
}

def get_suggested_prompt_type():
    return PROMPT_TEMPLATES


def data_to_prompt(data, prompt_type='translation_paragraph'):
    # 데이터 형태를 모델이 학습 가능한 프롬프트 형식으로 변환하는 함수
    if prompt_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")

    # 해당 prompt_type에 맞는 프롬프트 템플릿 가져오기
    prompt_template = PROMPT_TEMPLATES[prompt_type]
    
    processed_data = []
    for item in data:
        if 'input' in item and 'output' in item:
            input_text = prompt_template.format(input=item['input'])
            output_text = item['output']
            processed_data.append({'input': input_text, 'output': output_text})
        else:
            raise ValueError("Each data item must contain 'input' and 'output' keys.")
    
    return processed_data

def merge_data(*datasets):
    # 여러 개의 데이터셋을 합치는 함수
    merged_data = []
    for dataset in datasets:
        merged_data.extend(dataset)
    return merged_data




class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.trg = []
        self.label = []

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
        answer_dict = {
            "": None,
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat.append(f"화자{speaker}: {utterance}")
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화의 {inp['category']}"
            if (ord(inp['category'][-1]) - ord("가")) % 28 > 0:
                question += "으로"
            else:
                question = "로"
            question += " 올바른 지문은?"
                
            chat = chat + "\n\n" + question + "\n\n[Option]\n"
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target = ""
            if example["output"] == "inference_1":
                target = f"A. {example['input']['inference_1']}{tokenizer.eos_token}"
            elif example["output"] == "inference_2":
                target = f"B. {example['input']['inference_2']}{tokenizer.eos_token}"
            elif example["output"] == "inference_3":
                target = f"C. {example['input']['inference_3']}{tokenizer.eos_token}"
                
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.trg.append(answer_dict[example["output"]])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.trg[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )