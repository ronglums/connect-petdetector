
import shutil, tarfile, os, shutil, re
from os.path import basename, isfile
from urllib.parse import urlparse
from urllib.request import urlopen
from pathlib import Path

# Fetch a file from uri, unzip and untar it into its own directory.
def fetch_and_untar(uri):
    # Parse the uri to extract the local filename
    parsed_uri = urlparse(uri)
    local_filename = basename(parsed_uri.path)

    # If file is not already on disk, retrieve from uri
    if not isfile(local_filename):
        with urlopen(uri) as response:
            with open(local_filename, 'bw+') as f:
                shutil.copyfileobj(response, f)

    # Expand the archive
    with tarfile.open(local_filename) as tar:
        tar.extractall()
        
# The script below will rearrange the files so that all of the photos for a specific breed of dog 
# or cat will be stored in its own directory, where the name of the directory is the name of the
# pet's breed.
def move_images_into_labelled_directories(image_dir):
    images_path = Path(image_dir)
    extract_breed_from_filename = re.compile(r'([^/]+)_\d+.jpg$')

    for filename in os.listdir('images'):
        print(filename)
        match = extract_breed_from_filename.match(filename)
        if match is not None:
            breed = match.group(1)
            if not os.path.exists(images_path / breed):
                os.makedirs(images_path / breed)
            src_path = images_path / filename
            dest_path = images_path / breed / filename
            shutil.move(src_path, dest_path)
            
