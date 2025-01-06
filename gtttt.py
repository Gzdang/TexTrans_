from PIL import Image

ref_img = Image.open("dataset/chair/22/texture.png").convert("RGB").save("ref_texture.png")