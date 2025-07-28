from Qwen_model import Qwen_model
from embedder import Embidder

LLM_model = Qwen_model()
embedder = Embidder()


LLM_model.set_system_role()


book_path ='eBook-How-to-Build-a-Career-in-AI.pdf'
embedder.add_pdf(book_path, chunck_size=500)

question = "how can I start my career as AI Engineer?"
chuncks = embedder.retrieve_context(question, k=2)
print(chuncks[0]+"\n\n"+chuncks[1])
# answer = LLM_model.generate_RAG_response(chuncks,question)

# print(answer)
