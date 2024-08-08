import gradio as gr

def create_ui(generate_response, clear_history):

    ui = gr.Blocks()
    with ui:
        PATH_TO_CSS = "/Users/naishaagarwal/Documents/GitHub/UCLA-Class-Counselor/followup_project/rag/gradio_ui/style.css"
        PATH_TO_LOGO = "/Users/naishaagarwal/Documents/GitHub/UCLA-Class-Counselor/followup_project/rag/gradio_ui/images/ucla_logo.jpg"

        with open(PATH_TO_CSS) as css_file:
            css_content = css_file.read()

        gr.HTML(f"<style>{css_content}</style>")
        
        gr.Image(value= PATH_TO_LOGO, elem_id="logo", show_label=False)
        gr.Markdown("<div id='title'>UCLA Class Counselor</div>")
        with gr.Row():
            with gr.Column(scale=8):
                query = gr.Textbox(placeholder="Hi! I am your personal UCLA Class Counselor. Ask me anything about class offerings at UCLA!", lines=2, elem_id="textbox")
            with gr.Column(scale=2):
                button = gr.Button("Submit", elem_id="button")
                clear_button = gr.Button("Clear History", elem_id="clear")
        chatbot = gr.Chatbot()
        state = gr.State([])
        button.click(generate_response, inputs=[query, state], outputs=[chatbot, state])
        clear_button.click(clear_history, outputs=[chatbot, state])



    return ui

