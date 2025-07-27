from Email_fetch import Email
from Qwen_model import Qwen_model
import gradio as gr

email_fetch = Email()
model = Qwen_model()

email_fetch.get_latest_n_emails()


model.set_system_role(system_message="""You are a concise and helpful assistant. Summarize the following email in as few words as possible. 
Do not include or mention any hyperlinks or URLs in the summary.

Example of content to ignore:
https://info.deeplearning.ai/e3t/Ctc/...
                      """)


counter = 0
def next_email():
    global counter
    counter += 1
    sender , subject , body = email_fetch.find_next_email(n=counter ,max_char = 1000)
    response = model.run_query(body)
    output = f"from: {sender}\n" + response 
    return output


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("Check My E-mail AI")

    with gr.Row():

        output_text = gr.Textbox(label="summarization", scale=3)
        next_email_button = gr.Button("Next E-mail", scale=1)

    next_email_button.click(
        fn=next_email,
        inputs=[],
        outputs=output_text
    )

demo.launch()