import os
from PIL import Image, ImageDraw
import random

print('Creating sample images...')
os.makedirs('data/helipad', exist_ok=True)
os.makedirs('data/no_helipad', exist_ok=True)

def create_helipad_image(i):
    img = Image.new('RGB', (224, 224), color=(34, 139, 34))
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = 112, 112
    radius = 50
    draw.ellipse([center_x - radius, center_y - radius, 
                  center_x + radius, center_y + radius], 
                 fill=(128, 128, 128))
    
    # H marking
    draw.rectangle([center_x - 10, center_y - 20, center_x - 5, center_y + 20], fill=(255, 255, 255))
    draw.rectangle([center_x + 5, center_y - 20, center_x + 10, center_y + 20], fill=(255, 255, 255))
    draw.rectangle([center_x - 10, center_y - 2, center_x + 10, center_y + 2], fill=(255, 255, 255))
    
    img.save(f'data/helipad/helipad_{i:03d}.jpg')

def create_no_helipad_image(i):
    colors = [(34, 139, 34), (139, 69, 19), (70, 130, 180)]
    img = Image.new('RGB', (224, 224), color=random.choice(colors))
    draw = ImageDraw.Draw(img)
    
    for _ in range(random.randint(2, 4)):
        x1, y1 = random.randint(0, 112), random.randint(0, 112)
        x2, y2 = random.randint(112, 224), random.randint(112, 224)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    img.save(f'data/no_helipad/no_helipad_{i:03d}.jpg')

for i in range(50):
    create_helipad_image(i)
    create_no_helipad_image(i)

print('Created 50 helipad images and 50 no-helipad images!')
