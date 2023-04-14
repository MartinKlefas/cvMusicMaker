import numpy as np
from tqdm import tqdm

import kmeans_preprocessor

from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.applications.vgg16 import preprocess_input 
from keras.layers import Input

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from concurrent.futures import ThreadPoolExecutor

import pathlib, gc, pickle

def cluster(rootFolder : pathlib.Path = pathlib.Path("/"), principal_components : int = 2, clusters : int = 2, imageSize : int = 224):

    input_shape = (imageSize, imageSize, 3)
    input_layer = Input(shape=input_shape)

    model_ft = VGG16()
    model_ft = Model(inputs= model_ft.inputs,outputs = model_ft.layers[-2].output)

    files = rootFolder.rglob('*s.png')

    # we again need to use our ugly hack to do a progress bar, this time though we need a list of strings for cv2 to be able to open them
    fileNames = [str(p) for p in files]

    batch_size = 2000
    num_batches = int(np.ceil(len(fileNames) / batch_size))

    features = []
    for i in tqdm(range(num_batches),desc="Processing image batches"):
            start = i * batch_size
            end = (i + 1) * batch_size

            batch_fileNames = fileNames[start:end]

            with ThreadPoolExecutor() as executor:
                images = list(executor.map(kmeans_preprocessor.cvload_image, batch_fileNames, [imageSize]*len(batch_fileNames)))

            #print("Concatenating")
            images = np.concatenate(images, axis=0)
            
            #print("Pre-procssing")
            x = preprocess_input(images)
            del images
            gc.collect()

            #print("extracting features")
            batch_features = model_ft.predict(x, use_multiprocessing=True, verbose=0)
            
            
            features.append(batch_features.reshape(-1, 4096))

    
    
    print("concatenating features")
    features = np.concatenate(features, axis=0)
    

    del x
    gc.collect()

    print("PCA")
    pca = PCA(n_components=principal_components, random_state=22)
    pca.fit(features)
    x = pca.transform(features)


    del features
    gc.collect()

    print("fitting to kmeans")
    kmeans = KMeans(n_clusters=clusters, random_state=22,n_init="auto")
    kmeans.fit(x)
    
    print("preparing groups")
    groups = {}
    for file, cluster in zip(fileNames,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    return groups

imagePath = pathlib.Path("images/")

for directory in [x for x in imagePath.iterdir() if x.is_dir()]:
    print(f"testing {directory} for changes.")
    if not pathlib.Path(directory / "groups.pickle").exists() or kmeans_preprocessor.needs_updating(directory):
        print(f"processing {directory}")

        groups = cluster(rootFolder = directory, principal_components  = 2, clusters = 2)#, imageSize=112) 

        with open(str(directory / "groups.pickle"), 'wb') as handle:
            pickle.dump(groups, handle, protocol=pickle.HIGHEST_PROTOCOL)