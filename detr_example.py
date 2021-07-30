import matplotlib
import numpy as np
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import matplotlib.pyplot as plt
import torch
import cv2
matplotlib.use('Agg')

input_file = 'riviera.mp4'
output_file = 'riviera_out.mp4'
prob_threshold = 0.7


def write_frame(pil_img, prob, boxes, out):
    px = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(pil_img.shape[1] * px, pil_img.shape[0] * px), frameon=False)
    plt.imshow(pil_img)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax = plt.gca()
    for p, (x_min, y_min, x_max, y_max), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(x_min, y_min, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    out.write(img)
    plt.close('all')


batch_size = 5
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101').cuda()
cap = cv2.VideoCapture(input_file)
success, image = cap.read()
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (image.shape[1], image.shape[0]))
batch = []
i = 0
while success:
    while len(batch) < batch_size and success:
        success, image = cap.read()
        if success:
            batch.append(image)
    if not batch:
        break
    print(f'Batch: {i} size {len(batch)}')
    encoding = feature_extractor(batch[:], return_tensors='pt')
    outputs = model(pixel_values=encoding.data['pixel_values'].cuda(), pixel_mask=encoding.data['pixel_mask'].cuda())
    for attr, value in outputs.__dict__.items():
        if not attr.startswith('__') and value is not None:
            setattr(outputs, attr, value.cpu())
    probabilities = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probabilities.max(-1).values > prob_threshold
    post_processed = feature_extractor.post_process(outputs, torch.tensor([[batch[0].shape[0], batch[0].shape[1]]] * len(batch)))
    for img in batch:
        write_frame(img, probabilities[keep], post_processed[0]['boxes'][keep], out)
    batch = []
    i += 1

cap.release()
out.release()
cv2.destroyAllWindows()
