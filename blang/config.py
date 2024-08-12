

saved_model_info = {
	"base_models": {
		'MLP-KTLim/llama-3-Korean-Bllossom-8B': {
			"adapter_list": ['summarition', 'chatbot', 'relation_extraction', 'ner', 'santiment_analysis']
		}, 
		'MLP-KTLim/llama-3-Korean-Bllossom-70B': {
			"adapter_list": ['summarition', 'chatbot', 'relation_extraction', 'ner']
		}
	}
}


def get_model_list():
	return list(saved_model_info['base_models'].keys())



def get_adapter_list(base_model_name):
	return saved_model_info['base_models'][base_model_name]['adapter_list']