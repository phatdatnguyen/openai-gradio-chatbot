import os
import io
import base64
import gradio as gr
from openai import OpenAI
import PyPDF2
from docx import Document
import pandas as pd
from pptx import Presentation
import json
from api_key import API_KEY

client = OpenAI(
    api_key = API_KEY
)

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_image(llm_model, temperature, top_p, text, image, history):
    history = history or []
    history.append({"role": "user", "content": text})

    print(f'User: {text}\nImage: {image}')

    # Getting the base64 string
    base64_image = encode_image(image)

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

def read_text_file(file_path):
    with open(file_path, 'tr') as file:
        text = file.read()
    return text

def process_document(llm_model, temperature, top_p, text, document, history):
    document_text = ""
    try:
        _, file_extension = os.path.splitext(document)
        if file_extension.lower() == ".pdf":
            document_text = read_pdf_file(document)
        elif file_extension.lower() == ".docx":
            document_text = read_word_file(document)
        elif file_extension.lower() == ".xlsx":
            document_text = read_excel_file(document)
        elif file_extension.lower() == ".pptx":
            document_text = read_powerpoint_file(document)
        else:
            document_text = read_text_file(document)
    except:
        gr.Warning("Cannot read this file!")
        document_text = ""
    
    if not document_text == "":
        history = history or []
        history.append({"role": "user", "content": text + '\n' + document_text})

        print(f'User: {text}\nDocument file: {os.path.basename(document)}')

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

    # API call
    response = client.images.generate(model=image_generation_model, prompt=text, n=n_images, size=image_size, quality=image_quality)
    if n_images == 1:
        ai_message = "Here is the image I generated:"
    else:
        ai_message = "Here are the images I generated:"
    for i in range(len(response.data)):
        image_url = response.data[i].url
        ai_message += f"\n![Image {i + 1}]({image_url})"
    history.append({"role": "assistant", "content": ai_message})

    print(f'AI: {ai_message}')

    return history

def process_text(llm_model, temperature, top_p, text, history):
    history = history or []
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
    if llm_model == "gpt-3.5-turbo":
       image_input = gr.Image(label="Upload an image", sources=["upload"], type="pil", value=None, interactive=False)
    else:
       image_input = gr.Image(label="Upload an image", sources=["upload"], type="pil", value=None, interactive=True)

    return image_input

def on_image_generation_model_change(image_generation_model):
    if (image_generation_model) == "dall-e-3":
        n_images = gr.Slider(label="Number of images", minimum=1, maximum=1, step=1, value=1, interactive=False)
        image_size = gr.Dropdown(label="Image size", value="1024x1024", choices=["1024x1024", "1792x1024", "1024x1792"])
        image_quality = gr.Dropdown(label="Image quality", value="standard", choices=["standard", "hd"], interactive=True)
    else:
        n_images = gr.Slider(label="Number of images", minimum=1, maximum=5, step=1, value=1, interactive=True)
        image_size = gr.Dropdown(label="Image size", value="1024x1024", choices=["256x256", "512x512", "1024x1024"])
        image_quality = gr.Dropdown(label="Image quality", value="standard", choices=["standard"], interactive=False)
    
    return n_images, image_size, image_quality

def on_user_input(llm_model, temperature, top_p, text, image, document, history, generate_image, image_generation_model, n_images, image_size, image_quality):
    try:
        if image:
            history = process_image(llm_model, temperature, top_p, text, image, history)
        elif document:
            history = process_document(llm_model, temperature, top_p, text, document, history)
        elif generate_image:
            history = process_image_generation(image_generation_model, text, history, n_images, image_size, image_quality)
        elif text:
            history = process_text(llm_model, temperature, top_p, text, history)
            
        text_input = gr.Textbox(label="Message", placeholder="Type a message or question...", value=None)
        image_input = gr.Image(label="Upload an image", sources=["upload"], type="pil", value=None)
        document_input = gr.File(label="Upload a document", type="filepath", value=None)
        generate_image = gr.Checkbox(label="Generate image", value=False)
        return history, history, text_input, image_input, document_input, generate_image
    except Exception as exc:
        gr.Warning(str(exc.args))
        return history, history, text_input, image_input, document_input, generate_image

def save_history(history, history_file_name):
    try:
        if not history:
            raise Exception("No history")
        
        os.makedirs("history", exist_ok=True)
        with open(f"./history/{history_file_name}.json", "w") as file:
            json.dump(history, file, indent=4)
        return "Chat history saved successfully!"
    except Exception as exc:
        return f"Error saving history: {str(exc)}"

def load_history(saved_history_file):
    try:
        if not saved_history_file:
            raise Exception("No history file")
        
        with open(saved_history_file, "r") as file:
            history = json.load(file)
        return history, history, "Chat history loaded successfully!"
    except Exception as exc:
        return [], [], f"Error loading history: {str(exc)}"
    
with gr.Blocks() as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(type="messages", show_copy_button=True)
            state = gr.State([])
        with gr.Column(scale=1):
            history_file_name = gr.Textbox(label="File name", value="Chat history")    
            save_button = gr.Button(value="Save chat history")
            save_status = gr.Markdown(value="")
            saved_history_file = gr.File(label="Upload a chat history file", type="filepath", file_types=[".json"])
            load_button = gr.Button(value="Load chat history")
            load_status = gr.Markdown(value="")
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="Message", placeholder="Type a message or question...")
            llm_model = gr.Dropdown(label="Model", value="gpt-4o-mini", choices=[
                "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "chatgpt-4o-latest",  "o1-preview", "o1-mini"])
            temperature = gr.Slider(label="Temperature", minimum=0, maximum=2, step=0.01, value=1)
            top_p = gr.Slider(label="Top-p", minimum=0, maximum=1, step=0.01, value=1)
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload an image", sources=["upload"], type="pil")
            document_input = gr.File(label="Upload a document", type="filepath")
        with gr.Column(scale=1):
            generate_image = gr.Checkbox(label="Generate image", value=False)
            image_generation_model = gr.Dropdown(label="Model", value="dall-e-3", choices=["dall-e-3", "dall-e-2"])
            n_images = gr.Slider(label="Number of images", minimum=1, maximum=1, step=1, value=1, interactive=False)
            image_size = gr.Dropdown(label="Image size", value="1024x1024", choices=["1024x1024", "1792x1024", "1024x1792"])
            image_quality = gr.Dropdown(label="Image quality", value="standard", choices=["standard", "hd"])

    llm_model.change(on_llm_model_change, llm_model, image_input)
    image_generation_model.change(on_image_generation_model_change, image_generation_model, [n_images, image_size, image_quality])
    text_input.submit(on_user_input, [llm_model, temperature, top_p, text_input, image_input, document_input, state, generate_image, image_generation_model, n_images, image_size, image_quality], [chatbot, state, text_input, image_input, document_input, generate_image])
    
    save_button.click(save_history, [state, history_file_name], save_status)
    load_button.click(load_history, saved_history_file, [state, chatbot, load_status])

demo.launch(max_file_size=5*gr.FileSize.MB)


