import pickle, pathlib,uuid

from tqdm import tqdm

def remove_empty_directories(folder: pathlib.Path):
    for subdir in folder.iterdir():
        if subdir.is_dir():
            remove_empty_directories(subdir)  # Recursively remove empty directories inside this directory
            try:
                subdir.rmdir()  # Remove empty directory
                
            except OSError as e:
                pass

files = pathlib.Path("images/").rglob('groups.pickle')

for pFile in files:

    with open(str(pFile), 'rb') as handle:
        groups = pickle.load(handle)

    folder = pFile.parent

    for i in range(len(groups)):
        groupPath = folder / f"group {i}" / ""
        pathlib.Path(groupPath).mkdir(parents=True, exist_ok=True)
        files = groups[i]
        
        for index, file in tqdm(enumerate(files), total = len(files), desc=f"Moving from {str(pFile)}, group {i}.",bar_format='{l_bar}{bar}| {percentage:3.0f}% {n}/{total} [{remaining}{postfix}]'):
            try:
                pathlib.Path(file).rename(groupPath / f"{uuid.uuid4()}.png" )
            except:
                pass

remove_empty_directories(pathlib.Path("images/"))