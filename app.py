import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import os
from src.model import ImageCaptioningModel
from src.dataset import Vocabulary
from src.inference import generate_caption_greedy
from src.attention_viz import plot_attention
import matplotlib.pyplot as plt

# Try to load vocab and model (Requires user to train first or download checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size = 256
vocab_size = 10000 # Dummy value, should be loaded from actual trained vocab
num_layers = 3
nhead = 8

# Note: In a real deployment, you would serialize and load the Vocabulary object.
# For the demo skeleton, we define the structure.
vocab = Vocabulary(freq_threshold=5)

model = ImageCaptioningModel(
    embed_size=embed_size, 
    vocab_size=vocab_size, 
    num_layers=num_layers, 
    nhead=nhead
).to(device)

if os.path.exists("checkpoints/best_model.pth"):
    # Allow loading even if vocab size mismatches just for structure demo
    try:
        model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
        model.eval()
        model_loaded = True
    except Exception as e:
        print(f"Failed to load model: {e}")
        model_loaded = False
else:
    print("No trained model found at checkpoints/best_model.pth")
    model_loaded = False

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image):
    if not model_loaded:
        return "Model not trained yet. Please run the notebook first to train the model.", None
        
    image_tensor = transform(image).unsqueeze(0)
    
    caption_words, attention_maps = generate_caption_greedy(
        model, image_tensor, vocab, max_length=50, device=device
    )
    
    caption_text = " ".join([word for word in caption_words if word not in ["<start>", "<end>", "<pad>"]])
    
    fig = plot_attention(image, caption_words, attention_maps)
    
    return caption_text, fig

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Textbox(label="Generated Caption"),
        gr.Plot(label="Attention Heatmaps")
    ],
    title="Multimodal Image Captioning with Transformer Decoder",
    description="Upload an image to generate a caption. The attention heatmaps show where the model was looking for each word."
)

if __name__ == "__main__":
    demo.launch()
