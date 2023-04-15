import cv2
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
    
    folderPath.mkdir(parents=True, exist_ok=True)

filesList = [p for p in Path("images/").rglob("*[!s].png")] # this lazily imports our generator as a list so we can use a progress bar.

# If we have a lot of files, or don't want to use a hack to get a progress bar we can leave it as a generator:
# filesList = Path("images/").rglob("*.png")
# this should be faster, use less memory, and will still show how many iterations we're doing a second, but not know how long remains.
# processing 1,000 files it's about 10% faster not to use the hack

for thisFile in tqdm(filesList):
        
            
            
            try:
                #image = cv2.imread(str(thisFile))
                #resized_frame = cv2.resize(image, (640, 480), interpolation = cv2.INTER_AREA)
                #cv2.imwrite(str(thisFile),resized_frame)

                npImage = np.asarray(Image.open(str(thisFile)))
                tiled_array = reshape_split(npImage,(64,48))
                
                new_folder = thisFile.parent /  thisFile.stem / ""

                
                validate_folder(new_folder)
                
                counter = 0
                for column in tiled_array:
                    for imageCell in column:
                        im = Image.fromarray(imageCell)
                        im.save( new_folder / f"{counter}_s.png")    
                        counter += 1    
            except Exception as ex:
                print(thisFile)
                print(ex)


