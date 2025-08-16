import gradio as gr
from pathlib import Path
from upload_utils import handle_file_upload
from chat import generate_response_from_combined_text
from state import choices_list
import asyncio
import sys

#Main method to display and use the tool
def main():
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    with gr.Blocks(title="UniChunk Q&A Chatbot") as demo:
        gr.Markdown("# UniChunk Q&A Chatbot")
        gr.Markdown("By Pixel Pixies - Track 9 GenAI Fit.Fest'25")

        #maintain current_file in Gradio state and reset chat history if selected_file != current_file
        current_file = gr.State(value=None)

        chatbot = gr.Chatbot(label="Chat History", height=400, show_copy_button=True, avatar_images=(None, "robot.png"))

        with gr.Row():
            file_selector = gr.Dropdown(label="Select File from available docs", choices=choices_list, value=None, interactive=True)

        with gr.Row():
            with gr.Column(scale=0.75):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Ask a question based on a selected file...",
                    container=False
                )
            with gr.Column(scale=0.25, min_width=0):
                upload_btn = gr.UploadButton("Upload", file_types=[".pdf", ".txt"])

        def handle_file(chatbot_history, file):
            if file is None:
                return chatbot_history, gr.update()

            file_name = Path(file.name).name
            upload_status = handle_file_upload(file, query="summary")

            if file_name not in choices_list:
                choices_list.append(file_name)

            # Clear chat history for new uploads - changed to avoid any cross contamination and excess token accumulation
            chatbot_history = [(f"Uploaded: {file_name}", upload_status)]


            return chatbot_history, gr.update(choices=choices_list, value=file_name)

        def handle_query(chatbot_history, query, selected_file, current_file_val):
            if not selected_file:
                chatbot_history.append((query, "Please upload and select a file first."))
                return chatbot_history, "", current_file_val
            
            # Reset chat history if selected file has changed
            if selected_file != current_file_val:
                chatbot_history = []

            chatbot_history, _ = generate_response_from_combined_text(query, selected_file, chatbot_history)
            return chatbot_history, "", selected_file

        txt.submit(
            handle_query,
            inputs=[chatbot, txt, file_selector, current_file],
            outputs=[chatbot, txt, current_file],
            queue=False
        )

        upload_btn.upload(
            handle_file,
            inputs=[chatbot, upload_btn],
            outputs=[chatbot, file_selector],
            queue=False
        )

    demo.launch()

if __name__ == "__main__":
    main()