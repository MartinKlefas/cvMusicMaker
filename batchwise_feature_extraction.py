import numpy as np
from tqdm import tqdm

import kmeans_preprocessor

from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.applications.vgg16 import preprocess_input 

from concurrent.futures import ThreadPoolExecutor

import pathlib, gc, pickle, sys, uuid

def feat_current(folder : pathlib.Path):
    files = folder.rglob('*s.png')
    latest_file = max(files, key=lambda p: p.lstat().st_mtime)
    newest_file_time = latest_file.lstat().st_mtime
    pickle_file_time = pathlib.Path(folder / "features_all.pickle").lstat().st_mtime

    return pickle_file_time > newest_file_time

def batches_current(folder : pathlib.Path):
    files = folder.rglob('*s.png')
    latest_image = max(files, key=lambda p: p.lstat().st_mtime)

    newest_image_time = latest_image.lstat().st_mtime

    files = folder.rglob('features_batch_*.pickle')
    latest_pickle = max(files, key=lambda p: p.lstat().st_mtime)

    newest_pickle_time = latest_pickle.lstat().st_mtime

    return newest_pickle_time > newest_image_time


def feat_extract(rootFolder : pathlib.Path = pathlib.Path("/")):

    new_pickles_folder = pathlib.Path(rootFolder / "pickles" / str(uuid.uuid4()) / "")
    new_pickles_folder.mkdir(parents=True, exist_ok=True)

    model_ft = VGG16()
    model_ft = Model(inputs= model_ft.inputs,outputs = model_ft.layers[-2].output)

    files = rootFolder.rglob('*s.png')

    # we again need to use our ugly hack to do a progress bar, this time though we need a list of strings for cv2 to be able to open them
    fileNames = [str(p) for p in files]

    batch_size = 2000
    num_batches = int(np.ceil(len(fileNames) / batch_size))

    

    if pathlib.Path(rootFolder / "features_all.pickle").exists() and not feat_current(rootFolder):
          print("Everything seems to be up to date. Bye!")
          sys.exit()

    doneFilenames = []
    features = []

    if pathlib.Path(rootFolder / "features_batch_1.pickle").exists() and batches_current(rootFolder):
        print("reading old progress")
        feat_pickles = rootFolder.rglob('features*.pickle')

        

        for thisPickle in feat_pickles:
            with open(str(thisPickle), 'rb') as handle:
                    batch_features = pickle.load(handle)
            features.append(batch_features.reshape(-1, 4096))

        filename_pickles = rootFolder.rglob('filenames*.pickle')
        
        
        for thisPickle in filename_pickles:
            with open(str(thisPickle), 'rb') as handle:
                    batch_fileNames = pickle.load(handle)
            doneFilenames = doneFilenames + batch_fileNames

        #Change list to set, this is because checking membership in a set is much faster (O(1) complexity) than in a list (O(n) complexity)
        doneFilenames = set(doneFilenames)
        print(f"loaded progress for {len(doneFilenames)} files")
        
    


    for i in range(num_batches): #tqdm(range(num_batches),desc="Processing image batches"):
            print(f"Batch {i+1} of {num_batches+1}")
            start = i * batch_size
            end = (i + 1) * batch_size

            batch_fileNames = fileNames[start:end]
            if len(doneFilenames) > 0:
                batch_fileNames = [filename for filename in batch_fileNames if filename not in doneFilenames]

            if len(batch_fileNames) > 0:
                with ThreadPoolExecutor() as executor:
                    images = list(executor.map(kmeans_preprocessor.cvload_image, batch_fileNames, [224]*len(batch_fileNames)))

                #print("Concatenating")
                images = np.concatenate(images, axis=0)
                
                #print("Pre-procssing")
                x = preprocess_input(images)
                del images
                gc.collect()

                #print("extracting features")
                batch_features = model_ft.predict(x, use_multiprocessing=True, verbose=1)
                
                with open(str(new_pickles_folder/ f"features_batch_{i}.pickle"), 'wb') as handle:
                    pickle.dump(batch_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(str(new_pickles_folder/ f"filenames_batch_{i}.pickle"), 'wb') as handle:
                    pickle.dump(batch_fileNames, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                features.append(batch_features.reshape(-1, 4096))   

    print("concatenating features")
    features = np.concatenate(features, axis=0)

    with open(str(rootFolder/ "features_all.pickle"), 'wb') as handle:
                pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)


imagePath = pathlib.Path("images/")

for directory in [x for x in imagePath.iterdir() if x.is_dir()]:
    print(f"testing {directory} for changes.")
    feat_extract(directory)