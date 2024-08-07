

saved_model_info = {
	"base_models": {
		'bllossom_8b': {
			"adapter_list": ['summarition', 'chatbot', 'relation_extraction', 'ner', 'santiment_analysis']
		}, 
		'llama3_8b': {
			"adapter_list": ['summarition', 'chatbot', 'relation_extraction', 'ner']
		}
	}
}


def get_model_list():
	return list(saved_model_info['base_models'].keys())



def get_model_list(base_model_name):
	return saved_model_info['base_models'][base_model_name]['adapter_list']