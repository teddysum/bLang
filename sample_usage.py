import blang


blang.get_model_list()

model_name = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'

blang.get_adapter_list('MLP-KTLim/llama-3-Korean-Bllossom-8B')

# adapter_name = 'summarization'

model = blang.get_model(base_model=model_name, is_train=False)
# 학습용, 추론용(vLLM)

# model.get_suggested_prompt()
# model.set_prompt()

print(model.generate('안녕하세요'))
# model.inference('안녕하세요')

# model.train_adapter()

# model.save_adapter('./my_adapter')

# blang.evaluation(model)




# rag_database_address = ""
# rag_database_type = ""
# rag_embedding_model = ""
# rag_
# rag_system = blang.get_rag_system()
