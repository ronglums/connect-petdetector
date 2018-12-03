
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
#import matplotlib.style as style


def get_sample_images_for_each_species(dirname):
    d = Path(dirname)
    species_dirs = [d for d in d.iterdir() if d.is_dir()]
    species_images_and_labels = []
    for species_dir in species_dirs:
        for image_path in species_dir.iterdir():
            image = Image.open(image_path)
            image.thumbnail((224, 224), Image.ANTIALIAS)
            image_label = species_dir.parts[-1].lower().replace('_', ' ')
            species_images_and_labels.append((image, image_label))
            break
    return species_images_and_labels

def plot_images_in_grid(images_data, number_columns):
    f, subplots = plt.subplots(len(images_data) // number_columns + 1, number_columns)
    f.set_size_inches(16, 16)

    row = 0
    col = 0

    for record in images_data:
        subplot = subplots[row, col]
        subplot.imshow(record[0])
        subplot.set_axis_off()
        subplot.set_title(record[1], color='#358CD6')
        col += 1
        if col == number_columns:
            row += 1
            col = 0

    for c in range(col, number_columns):
        subplots[row, c].set_axis_off()
