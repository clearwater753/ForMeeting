from PIL import Image

from transformers import CLIPProcessor, CLIPModel
# # "openai/clip-vit-base-patch32"
path = r'D:\Desktop\lee_code-hot100\多模态\clip'
model = CLIPModel.from_pretrained(path)
processor = CLIPProcessor.from_pretrained(path)

image_path = r'D:\Desktop\lee_code-hot100\多模态\clip\000000039769.jpg'
image = Image.open(image_path)
# image.show()
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print("Label probs:", probs)