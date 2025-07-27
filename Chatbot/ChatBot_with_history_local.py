from Qwen_model import Qwen_model
import gradio as gr


model= Qwen_model()
model.set_system_role()

# while True:
#     query = input("You: ")
#     if query == "exit":
#         break
#     elif query == "clear":
#         model.delete_chat_history()

#     response = model.run_query(query)
#     model.save_chat_history(response)
#     print(response)



demo = gr.Interface(
    fn=model.run_from_app,
    inputs = "text",
    outputs = "text" 
)

demo.launch()