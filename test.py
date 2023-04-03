import cv2, os
from PIL import Image
from tqdm import tqdm
import numpy as np

def reshape_split(image: np.ndarray, kernel_size: tuple): 
    img_height, img_width, channels = image.shape 
    tile_width, tile_height = kernel_size 
    tiled_array = image.reshape(img_height // tile_height, tile_height, img_width // tile_width, tile_width, channels)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array 



npImage = np.asarray(Image.open("images/crab/0_r.jpg"))
tiled_array = reshape_split(npImage,(128,96))
#print(tiled_array.shape)


counter =0
new_folder = "images/crab/test/"

for column in tiled_array:
    for imageCell in column:
        im = Image.fromarray(imageCell)
        im.save(os.path.join(new_folder,f"{counter}.jpg"))        
        counter += 1

#counter = 0
#
#for tile in tiled_array:
#    cv2.imwrite(os.path.join(new_folder,f"{counter}.jpg"),tile)
#    counter += 1

