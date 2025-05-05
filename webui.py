import os
import io
import json
import glob
import re
import requests
import urllib3
import html2text
from PIL import Image
import base64
import gradio as gr
from openai import OpenAI
import PyPDF2
from docx import Document
import pandas as pd
from pptx import Presentation
from api_key import API_KEY

client = OpenAI(
    api_key = API_KEY
)

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def encode_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
    
    # Open the image using Pillow
    image = Image.open(io.BytesIO(response.content))
    
    # Encode the image
    return encode_image(image)

def process_image(llm_model, temperature, top_p, text, image, history):
    # Getting the base64 string
    base64_image = encode_image(image)

    history = history or []
    history.append({"role": "user", "content": text, "image_url": f"data:image/jpeg;base64,{base64_image}"})

    print(f'User: {text}\nImage: {image}')

    # API call
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url":  f"data:image/jpeg;base64,{base64_image}"}
                    }]
            }
        ],
        temperature=temperature,
        top_p=top_p)

    ai_message = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": ai_message})

    print(f'AI: {ai_message}')

    return history

def read_pdf_file(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
            text += '\n'
    return text

def read_word_file(file_path):
    document = Document(file_path)
    text = []
    for paragraph in document.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

def read_excel_file(file_path):
    excel_data = pd.read_excel(file_path)
    return excel_data.to_csv(index=False)

def read_powerpoint_file(file_path):
    presentation = Presentation(file_path)
    text = []
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return '\n'.join(text)

def read_html_file(file_path):
    with open(file_path, 'tr', encoding="utf-8") as file:
        text = file.read()
    return html2text.html2text(text)

def read_text_file(file_path):
    with open(file_path, 'tr', encoding="utf-8") as file:
        text = file.read()
    return text

def replace_history_content(history):
    replaced_history = []
    for message in history:
        content = message["content"]

        # Replace document content with document file name
        pattern = r'(?s)<<<DOCUMENT_CONTENT>>>\nFile: (.*?)\n.*?<<<END_DOCUMENT>>>'
        def repl(match):
            file_name = match.group(1)
            return f'[File: {file_name}]'
        replaced_content = re.sub(pattern, repl, content)

        # Replace link content with URL
        pattern = r'(?s)<<<LINK_CONTENT>>>\nURL: (.*?)\n.*?<<<END_LINK>>>'
        def repl(match):
            url = match.group(1)
            return f'[URL: {url}]'
        replaced_content = re.sub(pattern, repl, replaced_content)

        # Add encoded image
        if "image_url" in message:
            image_url = message["image_url"]
            replaced_content += f'\n![Image]({image_url})'
        
        replaced_message = {
                "role": message["role"],
                "content": replaced_content
            }
        replaced_history.append(replaced_message)

    return replaced_history

def process_document(llm_model, temperature, top_p, text, document, history):
    document_text = ""
    try:
        _, file_extension = os.path.splitext(document)
        file_name = os.path.basename(document)
        if file_extension.lower() == ".pdf":
            document_text = read_pdf_file(document)
        elif file_extension.lower() == ".docx":
            document_text = read_word_file(document)
        elif file_extension.lower() == ".xlsx":
            document_text = read_excel_file(document)
        elif file_extension.lower() == ".pptx":
            document_text = read_powerpoint_file(document)
        elif file_extension.lower() == ".htm" or file_extension.lower() == ".html":
            document_text = read_html_file(document)
        else:
            document_text = read_text_file(document)

        document_text = f"<<<DOCUMENT_CONTENT>>>\nFile: {file_name}\n{document_text}\n<<<END_DOCUMENT>>>"
    except Exception as exc:
        gr.Warning("Cannot read this file!\n" + str(exc.args))
        document_text = ""
    
    if not document_text == "":
        history = history or []
        history.append({"role": "user", "content": text + '\n' + document_text})

        print(f'User: {text}\nUser uploaded file: {os.path.basename(document)}')

        # API call
        response = client.chat.completions.create(
            model=llm_model,
            messages=history,
            temperature=temperature,
            top_p=top_p
        )

        ai_message = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": ai_message})

        print(f'AI: {ai_message}')

    return history

def process_image_generation(image_generation_model, text, history, n_images, image_size, image_quality):
    history = history or []
    history.append({"role": "user", "content": text})

    print(f'User: {text}')

    prompt = ""
    for message in history:
        prompt += f'{message["role"]}: {message["content"]}\n'

    # API call
    if image_generation_model == "gpt-image-1":
        response = client.images.generate(model=image_generation_model, prompt=prompt, size=image_size, quality=image_quality)
        ai_message = "Here is the image I generated:"
        print(f'AI: {ai_message}')
        for i in range(len(response.data)):
            base64_image = response.data[i].b64_json
            ai_message += f"\n![Image {i + 1}](data:image/jpeg;base64,{base64_image})"
        history.append({"role": "assistant", "content": ai_message})
    elif image_generation_model == "dall-e-3":
        response = client.images.generate(model=image_generation_model, prompt=prompt, size=image_size, quality=image_quality)
        ai_message = "Here is the image I generated:"
        print(f'AI: {ai_message}')
        for i in range(len(response.data)):
            image_url = response.data[i].url
            base64_image = encode_image_from_url(image_url)
            ai_message += f"\n![Image {i + 1}](data:image/jpeg;base64,{base64_image})"
        history.append({"role": "assistant", "content": ai_message})
    else: # image_generation_model == "dall-e-2":
        response = client.images.generate(model=image_generation_model, prompt=prompt, n=n_images, size=image_size)
        if n_images == 1:
            ai_message = "Here is the image I generated:"
        else:
            ai_message = "Here are the images I generated:"
        print(f'AI: {ai_message}')
        for i in range(len(response.data)):
            image_url = response.data[i].url
            base64_image = encode_image_from_url(image_url)
            ai_message += f"\n![Image {i + 1}](data:image/jpeg;base64,{base64_image})"
        history.append({"role": "assistant", "content": ai_message})
        
    return history

def process_text(llm_model, temperature, top_p, text, url, history):
    history = history or []

    if url:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        http = urllib3.PoolManager()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        response = http.request("GET", url, headers=headers)
        if response.status == 200:
            html = response.data.decode('utf-8')
        else:
            html = f"Failed to retrieve the webpage. Status code: {response.status}"
            gr.Warning(html + '\nYou can try downloading the web page as a HTML file and loading it as a document input.')
        
        html_text = html2text.html2text(html)
        text += f'\n<<<LINK_CONTENT>>>\nURL: {url}\n{html_text}\n<<<END_LINK>>>'
    history.append({"role": "user", "content": text})

    print(f'User: {text}')

    # API call
    response = client.chat.completions.create(
        model=llm_model,
        messages=history,
        temperature=temperature,
        top_p=top_p
    )

    ai_message = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": ai_message})

    print(f'AI: {ai_message}')

    return history

def on_llm_model_change(llm_model):
    if llm_model in ["gpt-3.5-turbo", "o1-mini", "o3-mini"]: # these models do not have vision capabilities
       return gr.update(interactive=False)
    else:
       return gr.update(interactive=True)

def on_image_generation_model_change(image_generation_model):
    if (image_generation_model) == "gpt-image-1":
        return gr.update(interactive=False), gr.update(choices=["auto", "1024x1024", "1536x1024", "1024x1536"], value="auto"), gr.update(interactive=True, choices=["auto", "low", "medium", "high"], value="auto")
    if (image_generation_model) == "dall-e-3":
        return gr.update(interactive=False), gr.update(choices=["auto", "1024x1024", "1792x1024", "1024x1792"], value="1024x1024"), gr.update(interactive=True, choices=["standard", "hd"], value="standard")
    else:
        return gr.update(interactive=True), gr.update(choices=["auto", "256x256", "512x512", "1024x1024"], value="1024x1024"), gr.update(interactive=False, choices=["standard", "hd"], value="standard")

def on_user_input(llm_model, temperature, top_p, text, image, document, url, history, generate_image, image_generation_model, n_images, image_size, image_quality):
    try:
        if image:
            history = process_image(llm_model, temperature, top_p, text, image, history)
        elif document:
            history = process_document(llm_model, temperature, top_p, text, document, history)
        elif generate_image:
            history = process_image_generation(image_generation_model, text, history, n_images, image_size, image_quality)
        elif text:
            history = process_text(llm_model, temperature, top_p, text, url, history)
            
        text_input = gr.Textbox(label="Message", placeholder="Type a message or question...", autofocus=True, value=None)
        image_input = gr.Image(label="Upload an image", sources=["upload", "clipboard"], type="pil", value=None)
        document_input = gr.File(label="Upload a document", type="filepath", value=None)
        url_input = gr.Textbox(label="Link", value=None)
        generate_image = gr.Checkbox(label="Generate image", value=False)
        replaced_history = replace_history_content(history)
        return history, replaced_history, text_input, image_input, document_input, url_input, generate_image
    except Exception as exc:
        gr.Warning(str(exc.args))
        text_input = gr.Textbox(label="Message", placeholder="Type a message or question...", autofocus=True, value=None)
        image_input = gr.Image(label="Upload an image", sources=["upload", "clipboard"], type="pil", value=None)
        document_input = gr.File(label="Upload a document", type="filepath", value=None)
        url_input = gr.Textbox(label="Link", value=None)
        generate_image = gr.Checkbox(label="Generate image", value=False)
        replaced_history = replace_history_content(history)
        return history, replaced_history, text_input, image_input, document_input, url_input, generate_image

def on_new_chat_click():
    return [], []

def on_toggle_history_column(state):
    state = not state
    return gr.update(visible = state), state

def get_history_file_list():
    history_file_list = []
    for file in glob.glob('.\\history\\*.json', recursive=True):
        history_file_list.append(os.path.splitext(os.path.basename(file))[0])

    return history_file_list

def select_history_file(evt: gr.SelectData):
    row_index = evt.index[0]
    history_file = get_history_file_list()[row_index]
    return history_file

def save_history(history, history_file_name):
    try:
        if not history:
            raise Exception("No history")
        
        os.makedirs("history", exist_ok=True)
        with open(f"./history/{history_file_name}.json", "w", encoding='utf8') as file:
            json.dump(history, file, indent=4, ensure_ascii=False)
        return "Chat history saved successfully!", pd.DataFrame(get_history_file_list(), columns=["File name"])
    except Exception as exc:
        return f"Error saving history: {str(exc)}", pd.DataFrame(get_history_file_list(), columns=["File name"])

def load_history(history_file_name: gr.Textbox):
    try:
        history_file_path = f"./history/{history_file_name}.json"
        with open(history_file_path, "r", encoding='utf8') as file:
            history = json.load(file)
        return history, replace_history_content(history), "Chat history loaded successfully!"
    except Exception as exc:
        return [], [], f"Error loading history: {str(exc)}"

with gr.Blocks() as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(type="messages", show_copy_button=True)
            state = gr.State([])
            history_column_state = gr.State(True)
            with gr.Row(equal_height=True):
                new_chat_button = gr.Button(value="New chat")
                toggle_history_column_button = gr.Button(value="Toggle chat history")
        with gr.Column(scale=1) as history_column:
            history_files_list = gr.DataFrame(label="Chat history files", value=pd.DataFrame(get_history_file_list(), columns=["File name"]), max_height=260)
            history_file_name = gr.Textbox(label="File name", value="Chat history")
            load_button = gr.Button(value="Load chat history")
            load_status = gr.Markdown(value="")
            save_button = gr.Button(value="Save chat history")
            save_status = gr.Markdown(value="")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion(label="Prompt"):
                text_input = gr.Textbox(label="Message", placeholder="Type a message or question...", autofocus=True)
                llm_model = gr.Dropdown(label="Model", value="gpt-4.1-mini", choices=[
                    "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.5-preview", "gpt-4o", "gpt-4o-mini", "chatgpt-4o-latest", "o1", "o1-mini", "o3", "o3-mini", "o4-mini"])
                temperature = gr.Slider(label="Temperature", minimum=0, maximum=2, step=0.01, value=1)
                top_p = gr.Slider(label="Top-p", minimum=0, maximum=1, step=0.01, value=1)
        with gr.Column(scale=1):
            with gr.Accordion(label="Input"):
                image_input = gr.Image(label="Upload an image", sources=["upload", "clipboard"], type="pil")
                document_input = gr.File(label="Upload a document", type="filepath")
                url_input = gr.Textbox(label="Link")
        with gr.Column(scale=1):
            with gr.Accordion(label="Image generation"):
                generate_image = gr.Checkbox(label="Generate image", value=False)
                image_generation_model = gr.Dropdown(label="Model", value="gpt-image-1", choices=["gpt-image-1", "dall-e-3", "dall-e-2"])
                n_images = gr.Slider(label="Number of images", minimum=1, maximum=1, step=1, value=1, interactive=False)
                image_size = gr.Dropdown(label="Image size", value="auto", choices=["auto", "1024x1024", "1536x1024", "1024x1536"])
                image_quality = gr.Dropdown(label="Image quality", value="auto", choices=["auto", "low", "medium", "high"])

    new_chat_button.click(on_new_chat_click, [], [chatbot, state])
    toggle_history_column_button.click(on_toggle_history_column, history_column_state, [history_column, history_column_state])

    llm_model.change(on_llm_model_change, llm_model, image_input)
    image_generation_model.change(on_image_generation_model_change, image_generation_model, [n_images, image_size, image_quality])
    text_input.submit(on_user_input, [llm_model, temperature, top_p, text_input, image_input, document_input, url_input, state, generate_image, image_generation_model, n_images, image_size, image_quality],
                      [state, chatbot, text_input, image_input, document_input, url_input, generate_image])
    
    history_files_list.select(select_history_file, [], history_file_name)
    load_button.click(load_history, history_file_name, [state, chatbot, load_status])
    save_button.click(save_history, [state, history_file_name], [save_status, history_files_list])

demo.launch(max_file_size=100*gr.FileSize.MB)


