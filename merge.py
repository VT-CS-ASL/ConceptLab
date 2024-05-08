import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

def merge_images(folder_path, output_path=None, specfic_name=""):
    if output_path is None:
        output_path = os.path.basename(folder_path) + '.jpeg'

    folder_path = folder_path + '/images'
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpeg') and (specfic_name=="" or specfic_name in f)]
    images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))

    total_height = sum(Image.open(img).size[1] for img in images)
    max_width = max(Image.open(img).size[0] for img in images)
    text_width = 800

    new_img = Image.new('RGB', (max_width + text_width, total_height), "white")
    y_offset = 0
    for img_path in images:
        img = Image.open(img_path)
        new_img.paste(img, (text_width, y_offset))
        y_offset += img.height

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(new_img)
    ax.axis('off')

    y_offset = 0
    for img_path in images:
        img = Image.open(img_path)
        step_number = os.path.splitext(os.path.basename(img_path))[0].split('_')[0] + ' steps'
        img_height = img.size[1]
        text_y_position = y_offset + img_height / 2
        ax.text(5, text_y_position, step_number, fontsize=10, color='black', verticalalignment='center',
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='square,pad=0.5'))
        y_offset += img_height

    plt.savefig(output_path, bbox_inches='tight', dpi=500)
    #plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path> <filter name, such as step_images> [output_path]")
        sys.exit(1)

    folder_path = sys.argv[1]
    name = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    merge_images(folder_path, output_path, name)
