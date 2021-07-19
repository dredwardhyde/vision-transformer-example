import numpy as np
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
import matplotlib.pyplot as plt


def show_attention_info(images, title, filename):
    fig, axes = plt.subplots(12, 12, figsize=(17, 15))
    fig.suptitle(title, fontsize=24)
    for i in range(12):
        for j in range(12):
            axes[i][j].set_xticklabels([])
            axes[i][j].set_xticks([])
            axes[i][j].set_yticklabels([])
            axes[i][j].set_yticks([])
            axes[i][j].imshow(images(i, j))
    fig.subplots_adjust(left=0.017, right=1.0, top=0.937, bottom=0.033, wspace=0.026, hspace=0.03)
    fig.savefig(filename, dpi=250)
    plt.clf()


image = Image.open(requests.get('http://images.cocodataset.org/val2017/000000039769.jpg', stream=True).raw)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
inputs = feature_extractor(images=image, return_tensors="pt")
result = model(**inputs, output_attentions=True)

attention_matrices = lambda i, j: result.attentions[i][0, j].detach().cpu().numpy()
attention_heads = lambda i, j: result.attentions[i][0, j, 180, 1:].reshape((14, 14)).detach().cpu().numpy()

show_attention_info(attention_matrices, 'Attention Matrices', 'matrices.png')
show_attention_info(attention_heads, 'Visualization of Attention', 'attention.png')
