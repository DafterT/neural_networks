import os
import string

from PIL import Image, ImageDraw, ImageFont

letters = string.ascii_letters

os.makedirs('../../pythonProject15/dataset', exist_ok=True)
[os.makedirs(f'dataset/{letter}', exist_ok=True) for letter in letters]
font_files = os.listdir('fonts')
image_size = (28, 28)
text_color = (0, 0, 0)
background_color = (255, 255, 255)

font_types = {'ttf', 'fon', 'TTC', 'ttc', 'TTF', 'otf'}

for font_file in font_files:
    try:
        if font_file.split('.')[-1].lower() not in font_types:
            continue
        for letter in letters:
            image = Image.new('RGB', image_size, background_color)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(f'fonts/{font_file}', size=int(image_size[0] * 0.7))
            _, _, w, h = draw.textbbox((0, 0), letter, font=font)
            draw.text(((image_size[0] - w) / 2, (image_size[1] - h) / 2), letter, font=font, fill=text_color)
            image_filename = f'dataset/{letter}/{font_file[:-4] + ("_upper" if letter.isupper() else "_lower")}.png'
            image.save(image_filename)
            print(f'Saved: {image_filename}')
    except OSError:
        os.remove(f'fonts/{font_file}')
        print(f'[!] Error {font_file}')
