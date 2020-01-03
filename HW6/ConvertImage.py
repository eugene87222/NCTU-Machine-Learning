import sys
import numpy as np
from PIL import Image

def openImage(filename):
	img = Image.open(filename)
	img = np.array(img)
	return img

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'python3 {sys.argv[0]} <imagename>')
    else:
        img = openImage(f'{sys.argv[1]}.png')
        img = img.reshape((-1, 3))
        with open(f'{sys.argv[1]}', 'w', encoding='utf-8') as file:
            for pixel in img:
                for color in pixel:
                    file.write(f'{color} ')