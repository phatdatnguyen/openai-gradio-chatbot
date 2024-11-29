import os
import io
import base64
import gradio as gr
from openai import OpenAI
import PyPDF2
import json
from api_key import API_KEY

client = OpenAI(
    api_key = API_KEY
)

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_image(llm_model, text, image, history):
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
        ])

    ai_message = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": ai_message})

    print(f'AI: {ai_message}')

    return history

def process_pdf(llm_model, text, pdf, history):
    pdf_text = ""
    try:
        with open(pdf, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                pdf_text += page.extract_text()
                pdf_text += '\n'
    except Exception as e:
        pass
    
    history = history or []
    history.append({"role": "user", "content": text + '\n' + pdf_text})

    print(f'User: {text}\nPDF file: {os.path.basename(pdf)}')

    # API call
    response = client.chat.completions.create(
        model=llm_model,
        messages=history
    )

    ai_message = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": ai_message})

    print(f'AI: {ai_message}')

    return history

def process_image_generation(llm_model, text, history, n_images, image_size, image_quality):
    history = history or []
    history.append({"role": "user", "content": text})

    print(f'User: {text}')

    # API call
    response = client.images.generate(model=llm_model, prompt=text, n=n_images, size=image_size, quality=image_quality)
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

def process_text(llm_model, text, history):
    history = history or []
    history.append({"role": "user", "content": text})

    print(f'User: {text}')

    # API call
    response = client.chat.completions.create(
        model=llm_model,
        messages=history
    )

    ai_message = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": ai_message})

    print(f'AI: {ai_message}')

    return history

def on_image_generation_model_change(image_generation_model: gr.Dropdown):
    if (image_generation_model) == "dall-e-3":
        n_images = gr.Slider(label="Number of images", minimum=1, maximum=1, step=1, value=1, interactive=False)
        image_size = gr.Dropdown(label="Image size", value="1024x1024", choices=["1024x1024", "1792x1024", "1024x1792"])
        image_quality = gr.Dropdown(label="Image quality", value="standard", choices=["standard", "hd"], interactive=True)
    else:
        n_images = gr.Slider(label="Number of images", minimum=1, maximum=5, step=1, value=1, interactive=True)
        image_size = gr.Dropdown(label="Image size", value="1024x1024", choices=["256x256", "512x512", "1024x1024"])
        image_quality = gr.Dropdown(label="Image quality", value="standard", choices=["standard"], interactive=False)
    
    return n_images, image_size, image_quality

def on_user_input(llm_model, text, image, pdf, history, generate_image, image_generation_model, n_images, image_size, image_quality):
    try:
        if image:
            history = process_image(llm_model, text, image, history)
        elif pdf:
            history = process_pdf(llm_model, text, pdf, history)
        elif generate_image:
            history = process_image_generation(image_generation_model, text, history, n_images, image_size, image_quality)
        elif text:
            history = process_text(llm_model, text, history)
            
        text_input = gr.Textbox(
            placeholder="Type a message or question...",
            show_label=False,
            value="",
            lines=1
        )
        image_input = gr.Image(
            sources=["upload"],
            type="pil",
            label="Upload an image",
            value=None
        )
        pdf_input = gr.File(label="Upload a PDF", type="filepath", file_types=[".pdf"])
        generate_image = gr.Checkbox(label="Generate image", value=False)
        return history, history, text_input, image_input, pdf_input, generate_image
    except Exception as exc:
        gr.Warning(str(exc.args))
        return history, history, text_input, image_input, pdf_input, generate_image

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
            history_file_name = gr.Textbox(label="File name", lines=1, value="Chat history")    
            save_button = gr.Button(value="Save chat history")
            save_status = gr.Markdown(value="")
            saved_history_file = gr.File(label="Upload a chat history file", type="filepath", file_types=[".json"])
            load_button = gr.Button(value="Load chat history")
            load_status = gr.Markdown(value="")
    with gr.Row():
        with gr.Column(scale=1):
            llm_model = gr.Dropdown(label="Model", value="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o", "o1-preview", "o1-mini", "gpt-3.5-turbo"])
            text_input = gr.Textbox(label="Message", placeholder="Type a message or question...", lines=1)
        with gr.Column(scale=1):
            image_input = gr.Image(
                sources=["upload"],
                type="pil",
                label="Upload an image"
            )
            pdf_input = gr.File(label="Upload a PDF", type="filepath", file_types=[".pdf"])
        with gr.Column(scale=1):
            generate_image = gr.Checkbox(label="Generate image", value=False)
            image_generation_model = gr.Dropdown(label="Model", value="dall-e-3", choices=["dall-e-3", "dall-e-2"])
            n_images = gr.Slider(label="Number of images", minimum=1, maximum=1, step=1, value=1, interactive=False)
            image_size = gr.Dropdown(label="Image size", value="1024x1024", choices=["1024x1024", "1792x1024", "1024x1792"])
            image_quality = gr.Dropdown(label="Image quality", value="standard", choices=["standard", "hd"])

    image_generation_model.change(on_image_generation_model_change, image_generation_model, [n_images, image_size, image_quality])
    text_input.submit(on_user_input, [llm_model, text_input, image_input, pdf_input, state, generate_image, image_generation_model, n_images, image_size, image_quality], [chatbot, state, text_input, image_input, pdf_input, generate_image])
    
    save_button.click(save_history, [state, history_file_name], save_status)
    load_button.click(load_history, saved_history_file, [state, chatbot, load_status])

demo.launch()


