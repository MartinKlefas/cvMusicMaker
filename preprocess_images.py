import cv2, os
from PIL import Image
from tqdm import tqdm
import numpy as np

from pathlib import Path


def reshape_split(image: np.ndarray, kernel_size: tuple): 
    img_height, img_width, channels = image.shape 
    tile_width, tile_height = kernel_size 
    tiled_array = image.reshape(img_height // tile_height, tile_height, img_width // tile_width, tile_width, channels)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array 

def validate_folder(folderPath):
    
    Path(folderPath).mkdir(parents=True, exist_ok=True)

for dirname, _, filenames in os.walk("images/"):
        for filename in tqdm(filenames):
            
            thisFile = os.path.join(dirname, filename)
            image = cv2.imread(thisFile)
            resized_frame = cv2.resize(image, (640, 480), interpolation = cv2.INTER_AREA)
            cv2.imwrite(thisFile,resized_frame)

            npImage = np.asarray(Image.open(thisFile))
            tiled_array = reshape_split(npImage,(64,48))
            
            new_folder = os.path.join(dirname, filename,"")
            validate_folder(new_folder)
            
            counter = 0
            for column in tiled_array:
                for imageCell in column:
                    im = Image.fromarray(imageCell)
                    im.save(os.path.join(new_folder,f"{counter}.jpg"))    
                    counter += 1



