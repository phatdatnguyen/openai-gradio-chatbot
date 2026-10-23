"""Application shell: each chatbot owns its own layout, events, and state."""

import gradio as gr

from chatbot import config
from chatbot.chatbot import build_tab as build_chat_tab
from chatbot.log import setup_logging
from rag_chatbot.chatbot import build_tab as build_rag_tab
from plugins.ui import build_tab as build_plugins_tab


APP_CSS = """
.rag-source-checkboxes [data-testid="checkbox-group"] {
    max-height: 260px;
    overflow-y: auto;
}
.rag-source-checkboxes [data-testid="checkbox-group"] > label {
    width: 100%;
    box-sizing: border-box;
}
.rag-source-checkboxes [data-testid="checkbox-group"] > label > span {
    overflow-wrap: anywhere;
    min-width: 0;
}
.rag-summary-sidebar {
    justify-content: space-between;
}
.rag-summary-preview {
    overflow-y: auto;
}
.rag-quiz-options .wrap {
    flex-direction: column;
    flex-wrap: nowrap;
}
.rag-quiz-options .wrap > label {
    width: 100%;
    box-sizing: border-box;
}
.rag-quiz-options .wrap > label > span {
    overflow-wrap: anywhere;
    min-width: 0;
}
.rag-quiz-options input[type="radio"] {
    flex-shrink: 0;
}
"""


def create_app():
    with gr.Blocks(title="OpenAI Chatbots") as app:
        with gr.Tabs():
            with gr.Tab("Chat"):
                chat_tab = build_chat_tab()
            with gr.Tab("RAG Chat"):
                rag_tab = build_rag_tab()
            with gr.Tab("Plugins"):
                plugins_tab = build_plugins_tab()
        app.load(plugins_tab["refresh_callback"],
                 outputs=[plugins_tab["catalog"], plugins_tab["status"],
                          plugins_tab["snapshot"]], queue=False)
    app.chat_tab = chat_tab
    app.rag_tab = rag_tab
    app.plugins_tab = plugins_tab
    return app


demo = create_app()

if __name__ == "__main__":
    logger = setup_logging()
    config.check_model_tables()
    if config.client is None:
        logger.warning(config.MISSING_KEY_MESSAGE)
    demo.queue().launch(server_name="127.0.0.1", share=False,
                        max_file_size=100 * gr.FileSize.MB, css=APP_CSS)
