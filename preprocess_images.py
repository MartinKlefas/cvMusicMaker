import cv2, os
from tqdm import tqdm
import numpy as np

def reshape_split(image: np.ndarray, kernel_size: tuple): 
    img_height, img_width, channels = image.shape 
    tile_width, tile_height = kernel_size 
    tiled_array = image.reshape(img_height // tile_height, tile_height, img_width // tile_width, tile_width, channels)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array 

for dirname, _, filenames in os.walk("images/"):
        for filename in tqdm(filenames):
            
            thisFile = os.path.join(dirname, filename)
            image = cv2.imread(thisFile)
            resized_frame = cv2.resize(image, (640, 480), interpolation = cv2.INTER_AREA)
            cv2.imwrite(thisFile,resized_frame)
            npImage = np.asarray(resized_frame)
            tiled_array = reshape_split(npImage,(32,24))
            
            new_folder = os.path.join(dirname, filename,"")
            counter = 0
            for tile in tiled_array:
                 cv2.imwrite(os.path.join(new_folder,f"{counter}.jpg"))
                 counter += 1



