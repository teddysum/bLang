

model_configuration = {
	
}

class blang_model:
	def __init__(self, base_model, adapter):
		self.base_model = base_model
		self.adapter = adapter
		# model load and merge here
		self.merged_model = None

	def generate(self, input_sentence):
		# 실제 모델 생성 로직은 주석 처리되어 있습니다.
		# self.merged_model.generate(input_sentence)

		return f"input: {input_sentence}, generated result would be returned"


def get_model(base_model='bllossom_8b', adapter=None):

	print(f'get model based {base_model}')
	if adapter:
		print(f'adapter is {adapter}')
		print('get merged model')

	model = blang_model(base_model, adapter)

	return model