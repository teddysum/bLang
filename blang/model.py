import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq  # DataCollatorForSeq2Seq 추가
)
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset

model_configuration = {
    "temperature": 0.8, 
    "top_p": 0.95,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q_proj", "v_proj"],
    "learning_rate": 2e-5,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 0,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "save_steps": 100,
    "logging_steps": 10,
    "save_total_limit": 20,
    "weight_decay":0.1,
}

model_max_length = {
    'MLP-KTLim/llama-3-Korean-Bllossom-8B': 8192
}

class ModelDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.model_max_length = 2048
        print(tokenizer.model_max_length)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 입력 시퀀스와 출력 시퀀스를 토큰화하고 패딩 및 트렁케이션 적용
        inputs = self.tokenizer(
            item['input'],
            max_length=self.model_max_length // 2,  # 입력과 출력의 최대 길이를 반으로 나누어 설정
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = self.tokenizer(
            item['output'],
            max_length=self.model_max_length // 2,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # 첫 번째 차원 제거
            'attention_mask': inputs['attention_mask'].squeeze(0),  # 첫 번째 차원 제거
            'labels': labels['input_ids'].squeeze(0)  # 첫 번째 차원 제거
        }


class blang_model:
    def __init__(self, base_model, adapter, is_train, use_streaming):

        self.base_model_name = base_model
        self.adapter_name = adapter
        self.tokenizer = None
        # model load and merge here
        self.base_model = None
        self.is_train = is_train
        self.use_streaming = use_streaming
        self.adapter_path = None

        if self.is_train:
            # Training mode: Load the base model and prepare it for LoRA training
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            self.tokenizer.pad_token = self.tokenizer.eos_token


            # LoRA configuration
            lora_config = LoraConfig(
                r=model_configuration['lora_rank'],  # Rank of the factorization
                lora_alpha=model_configuration['lora_alpha'],  # Scaling factor
                lora_dropout=model_configuration['lora_dropout'],  # Dropout for LoRA layers
                target_modules=model_configuration['lora_target_modules']  # Target layers to apply LoRA
            )

            # Prepare model for k-bit training if necessary and apply LoRA configuration
            # self.base_model = prepare_model_for_kbit_training(self.base_model)
            self.base_model = get_peft_model(self.base_model, lora_config)

        else:
            # Inference mode: Load the model with vLLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            print('stop_token_ids: ', self.tokenizer.eos_token)
            self.sampling_params = SamplingParams(temperature=model_configuration['temperature'], top_p=model_configuration['top_p'], max_tokens=2000, stop_token_ids=[0])
            if self.adapter_name:
                # Download the LoRA adapter and initialize the model with LoRA enabled
                try:
                    self.adapter_path = snapshot_download(repo_id=self.adapter_name)
                except:
                    self.adapter_path  = self.adapter_name
                self.base_model = LLM(model=self.base_model_name, enable_lora=True)
            else:
                # Initialize the model without LoRA
                self.base_model = LLM(model=self.base_model_name, enable_lora=False)


    def generate(self, input_sentence):
        if self.is_train:
            raise ValueError("Generate method should only be used in inference mode.")

        if self.use_streaming:
            stream = self.base_model.generate_streaming(input_sentence, sampling_params=self.sampling_params)
            for chunk in stream:
                print(chunk)  # 각 스트리밍 결과를 출력합니다.
        else:
            if self.adapter_name:
                result = self.base_model.generate(input_sentence, sampling_params=self.sampling_params, lora_request=LoRARequest("translate_adapter", 1, self.adapter_path))
                return result
            else:
                result = self.base_model.generate(input_sentence, sampling_params=self.sampling_params)
                return result
            


    def inference(self, input_sentence):

        return self.generate(input_sentence)


    def train_adapter(self, train_data, save_path="./"):
        if not self.is_train:
            raise ValueError("Training can only be performed in training mode.")

        print(f'Training {self.base_model_name} model with LoRA layers.')

        training_args = TrainingArguments(
            output_dir=save_path,
            per_device_train_batch_size=model_configuration["per_device_train_batch_size"],
            num_train_epochs=model_configuration["num_train_epochs"],
            logging_dir='./logs',
            logging_steps=model_configuration["logging_steps"],
            save_steps=model_configuration["save_steps"],
            save_total_limit=model_configuration["save_total_limit"],
            learning_rate=model_configuration["learning_rate"],  # 학습률 설정
            gradient_accumulation_steps=model_configuration["gradient_accumulation_steps"],  # 그래디언트 누적 단계 설정
            warmup_steps=model_configuration["warmup_steps"],  # 워밍업 단계 설정
            weight_decay=model_configuration["weight_decay"],
            evaluation_strategy="no",  # 필요시 평가 전략 추가
            save_strategy="steps",  # 모델 저장 전략
            report_to="none"  # 필요시 wandb, tensorboard 등을 설정
        )
        
        train_dataset = ModelDataset(train_data, self.tokenizer)

        print(train_dataset)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.base_model
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.base_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        # Train the model with LoRA layers
        trainer.train()


    def save_adapter(self, save_path):
        if not self.is_train:
            raise ValueError("Saving adapters can only be done in training mode.")

        print(f'Saving {self.base_model_name} model with LoRA layers at {save_path}')
        self.base_model.save_pretrained(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)


def get_model(base_model='MLP-KTLim/llama-3-Korean-Bllossom-8B', adapter=None, is_train=True, use_streaming=False):

    print(f'get model based {base_model}')
    if adapter:
        print(f'adapter is {adapter}')
        print('get merged model')

    model = blang_model(base_model, adapter, is_train, use_streaming)

    return model