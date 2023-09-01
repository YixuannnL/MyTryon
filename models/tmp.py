from PIL import Image
import os

path = '/home/lyx/Code/Data/frameData/C0098/0010.jpg'
image = Image.open(path)

width, height = image.size

left = (width - 1080) // 2
top = (height - 1080) // 2

right = left + 1080
bottom = top + 1080

cropped_image = image.crop((left, top, right, bottom))

output_path = '/home/lyx/Code/Data/crop1.jpg'

cropped_image.save(output_path)