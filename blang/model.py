from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader


model_configuration = {
	"temperature": 0.8, 
	"top_p": 0.95,
	"lora_rank": 16,
	"lora_alpha": 32,
	"lora_dropout": 0.1,
	"lora_target_modules": ["q_proj", "v_proj"]
}

model_max_length = {
	'MLP-KTLim/llama-3-Korean-Bllossom-8B': 8192
}

class ModelDataset(Dataset):
    def __init__(self, data, tokenizer, model_max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.model_max_length = tokenizer.model_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 먼저 입력과 출력을 토큰화하여 길이를 확인
        input_tokens = self.tokenizer.encode(item['input'], add_special_tokens=False)
        output_tokens = self.tokenizer.encode(item['output'], add_special_tokens=False)

        # 전체 길이가 model_max_length를 넘지 않도록 조정
        total_length = len(input_tokens) + len(output_tokens)

        if total_length > self.model_max_length:
            # 길이 초과 시, 비율에 따라 입력과 출력 시퀀스의 길이를 조정
            input_max_length = int(self.model_max_length // 2)
            output_max_length = self.model_max_length - input_max_length

            input_ids = self.tokenizer.encode(item['input'], max_length=input_max_length, truncation=True)
            labels = self.tokenizer.encode(item['output'], max_length=output_max_length, truncation=True)
        else:
            input_ids = input_tokens
            labels = output_tokens

        # Padding 및 Tensor 변환
        input_ids = self.tokenizer.pad({'input_ids': input_ids}, padding='max_length', max_length=self.model_max_length, return_tensors='pt')['input_ids'].squeeze()
        labels = self.tokenizer.pad({'input_ids': labels}, padding='max_length', max_length=self.model_max_length, return_tensors='pt')['input_ids'].squeeze()

        return {'input_ids': input_ids, 'labels': labels}


class blang_model:
	def __init__(self, base_model, adapter, is_train, use_streaming):

		self.base_model_name = base_model
		self.adapter_name = adapter
		self.tokenizer = None
		# model load and merge here
		self.base_model = None
		self.is_train = is_train
		self.use_streaming = use_streaming

		if self.is_train:
			# Training mode: Load the base model and prepare it for LoRA training
			self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
			self.base_model = AutoModelForCausalLM.from_pretrained(
				self.base_model_name,
				device_map="auto"
			)

			# LoRA configuration
			lora_config = LoraConfig(
				r=model_configuration['lora_rank'],  # Rank of the factorization
				lora_alpha=model_configuration['lora_alpha'],  # Scaling factor
				lora_dropout=model_configuration['lora_dropout'],  # Dropout for LoRA layers
				target_modules=model_configuration['lora_target_modules']  # Target layers to apply LoRA
			)

			# Prepare model for k-bit training if necessary and apply LoRA configuration
			self.base_model = prepare_model_for_kbit_training(self.base_model)
			self.base_model = get_peft_model(self.base_model, lora_config)

		else:
			# Inference mode: Load the model with vLLM
			self.sampling_params = SamplingParams(temperature=model_configuration['temperature'], top_p=model_configuration['top_p'])
			if self.adapter_name:
				# Download the LoRA adapter and initialize the model with LoRA enabled
				adapter_path = snapshot_download(repo_id=self.adapter_name)
				self.base_model = LLM(model=self.base_model_name, enable_lora=True, lora_path=adapter_path)
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
			result = self.base_model.generate(input_sentence, sampling_params=self.sampling_params)
			return result


	def inference(self, input_sentence):

		return self.generate(input_sentence)


	def train_adapter(self, train_data, save_path="./"):
		if not self.is_train:
			raise ValueError("Training can only be performed in training mode.")

		print(f'Training {self.base_model_name} model with LoRA layers.')

		# Set up training arguments
		training_args = TrainingArguments(
			output_dir=save_path,
			per_device_train_batch_size=4,
			num_train_epochs=3,
			logging_dir='./logs',
			logging_steps=10,
			save_steps=1000,
			save_total_limit=3,
		)
		
		train_dataset = ModelDataset(train_data, self.tokenizer)
		train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)

		# Initialize Trainer
		trainer = Trainer(
			model=self.base_model,
			args=training_args,
			train_dataset=train_dataloader.dataset,
			tokenizer=self.tokenizer
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