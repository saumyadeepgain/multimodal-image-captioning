import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def plot_attention(image, result_caption, attention_plot):
    """
    Plots the attention maps overlaid on the original image for each generated word.
    
    Args:
        image: PIL Image
        result_caption: List of generated words (including <start> and <end>)
        attention_plot: List of attention maps corresponding to each word.
                        Each map is expected to be a 1D tensor of length 49 (for 7x7 grid).
    """
    # Remove <start> and <end> for visualization if desired, or keep them.
    # Usually, we visualize from the first generated word until <end>.
    temp_image = np.array(image)
    
    # Calculate grid size for subplots
    num_words = len(result_caption)
    cols = 4
    rows = math.ceil(num_words / cols)
    
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    
    for l in range(num_words):
        word = result_caption[l]
        if word == "<start>":
            continue
            
        # The attention plot list starts from the first generated word
        # (index l-1 if we skip <start>)
        attn_map = attention_plot[l - 1].cpu().numpy()
        
        # Reshape attention weights from 1D (49,) to 2D (7, 7)
        grid_size = int(math.sqrt(attn_map.shape[0]))
        attn_map = attn_map.reshape((grid_size, grid_size))
        
        ax = fig.add_subplot(rows, cols, l)
        ax.set_title(word, fontsize=16)
        
        # Show original image
        ax.imshow(temp_image)
        
        # Overlay attention map
        # We resize the 7x7 attention map to the image size using interpolation
        ax.imshow(attn_map, cmap='gray', alpha=0.6, extent=(0, temp_image.shape[1], temp_image.shape[0], 0), interpolation='bicubic')
        ax.axis('off')

    plt.tight_layout()
    return fig
