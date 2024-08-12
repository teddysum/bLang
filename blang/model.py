from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


model_configuration = {
	"temperature": 0.8, 
	"top_p": 0.95
}

class blang_model:
	def __init__(self, base_model, adapter, is_train):

		self.base_model_name = base_model
		self.adapter_name = adapter
		self.tokenizer = None
		# model load and merge here
		self.base_model = None
		self.is_train = is_train

		if self.is_train:
			# Training mode: Load the base model and prepare it for LoRA training
			self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
			self.base_model = AutoModelForCausalLM.from_pretrained(
				self.base_model_name,
				device_map="auto"
			)

			# LoRA configuration
			lora_config = LoraConfig(
				r=16,  # Rank of the factorization
				lora_alpha=32,  # Scaling factor
				lora_dropout=0.1,  # Dropout for LoRA layers
				target_modules=["q_proj", "v_proj"]  # Target layers to apply LoRA
			)

			# Prepare model for k-bit training if necessary and apply LoRA configuration
			self.base_model = prepare_model_for_kbit_training(self.base_model)
			self.base_model = get_peft_model(self.base_model, lora_config)

		else:
			# Inference mode: Load the model with vLLM
			sampling_params = SamplingParams(temperature=model_configuration['temperature'], top_p=model_configuration['top_p'])
			if self.adapter_name:
				# Download the LoRA adapter and initialize the model with LoRA enabled
				adapter_path = snapshot_download(repo_id=self.adapter_name)
				self.base_model = LLM(model=self.base_model_name, enable_lora=True, lora_path=adapter_path, sampling_params=sampling_params)
			else:
				# Initialize the model without LoRA
				self.base_model = LLM(model=self.base_model_name, enable_lora=False, sampling_params=sampling_params)


	def generate(self, input_sentence):
		if self.is_train:
			raise ValueError("Generate method should only be used in inference mode.")
		
		# Inference logic using the vLLM model
		result = self.base_model.generate(input_sentence)
		return result

	def inference(self, input_sentence):

		return self.generate(input_sentence)


	def train_adapter(self, train_data, output_dir):
		if not self.is_train:
			raise ValueError("Training can only be performed in training mode.")

		print(f'Training {self.base_model_name} model with LoRA layers.')

		# Set up training arguments
		training_args = TrainingArguments(
			output_dir=output_dir,
			per_device_train_batch_size=4,
			num_train_epochs=3,
			logging_dir='./logs',
			logging_steps=10,
			save_steps=1000,
			save_total_limit=3,
		)
		
		# Initialize Trainer
		trainer = Trainer(
			model=self.base_model,
			args=training_args,
			train_dataset=train_data,
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


def get_model(base_model='bllossom_8b', adapter=None, is_train=True):

	print(f'get model based {base_model}')
	if adapter:
		print(f'adapter is {adapter}')
		print('get merged model')

	model = blang_model(base_model, adapter, is_train)

	return model