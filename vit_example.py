import numpy as np
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn.functional as f


def show_attention_matrices(images, image_in_batch, title, filename):
    fig, axes = plt.subplots(12, 12, figsize=(17, 15))
    fig.suptitle(title, fontsize=24)
    for i in range(12):
        for j in range(12):
            axes[i][j].set_xticklabels([])
            axes[i][j].set_xticks([])
            axes[i][j].set_yticklabels([])
            axes[i][j].set_yticks([])
            axes[i][j].imshow(images(i, j, image_in_batch))
    fig.subplots_adjust(left=0.017, right=1.0, top=0.937, bottom=0.033, wspace=0.026, hspace=0.03)
    fig.savefig(filename + "_" + str(image_in_batch) + ".png", dpi=250)
    plt.clf()


def show_attention_heads(images, image_in_batch, title, filename):
    fig, axes = plt.subplots(12, 12, figsize=(17, 15))
    fig.suptitle(title, fontsize=24)
    for i in range(12):
        for j in range(12):
            axes[i][j].set_xticklabels([])
            axes[i][j].set_xticks([])
            axes[i][j].set_yticklabels([])
            axes[i][j].set_yticks([])
            axes[i][j].imshow(images(i, j, image_in_batch, 180))
    fig.subplots_adjust(left=0.017, right=1.0, top=0.937, bottom=0.033, wspace=0.026, hspace=0.03)
    fig.savefig(filename + "_" + str(image_in_batch) + " " + str(180) + ".png", dpi=250)
    plt.clf()


image_urls = ['https://i.pinimg.com/originals/b5/43/52/b54352f733dfd6e3023b1e12d0910d69.jpg',
              'http://images.cocodataset.org/val2017/000000039769.jpg']

images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k', do_resize=False)
model = ViTModel(ViTConfig(size=(4032, 3024)))
inputs = feature_extractor(images=images[0], return_tensors="pt")
result = model(**inputs, output_attentions=True)

embedding_1 = result.last_hidden_state[0][0, :]
embedding_2 = result.last_hidden_state[1][0, :]

attention_matrices = lambda i, j, z: result.attentions[i][z, j].detach().cpu().numpy()
attention_heads = lambda i, j, z, k: result.attentions[i][z, j, k, 1:].reshape((14, 14)).detach().cpu().numpy()

show_attention_matrices(attention_matrices, 0, 'Attention Matrices', 'matrices')
show_attention_heads(attention_heads, 0, 'Visualization of Attention', 'attention')

similarity_score = f.cosine_similarity(embedding_1.unsqueeze(0), embedding_2.unsqueeze(0)).item()
print(similarity_score)
