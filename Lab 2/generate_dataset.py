import os
import string

from PIL import Image, ImageDraw, ImageFont
from constants import image_size

letters = string.ascii_lowercase

os.makedirs('dataset', exist_ok=True)

text_color = (0, 0, 0)
background_color = (255, 255, 255)

font_file = 'times_new_roman.ttf'

for letter in letters:
    image = Image.new('RGB', image_size, background_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(f'{font_file}', size=int(image_size[0] * 0.8))
    _, _, w, h = draw.textbbox((0, 0), letter, font=font)
    draw.text(((image_size[0] - w) / 2, (image_size[1] - h) / 2), letter, font=font, fill=text_color)
    image_filename = f'dataset/{font_file[:-4]}_{letter}.png'
    image.save(image_filename)
    print(f'Saved: {image_filename}')
