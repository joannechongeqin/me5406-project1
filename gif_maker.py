import os
from PIL import Image
import imageio

def create_gif_from_folder(folder_path, gif_name, duration=5):
        filter_keyword = gif_name.split('_')[-1].split('.')[0]
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg')) 
                       and filter_keyword in f]
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
        images = []
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            img = Image.open(image_path)
            images.append(img)
        gif_path = os.path.join(folder_path, gif_name)
        if len(images) > 0:
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration*len(image_files), loop=0)
        print(f"\nGIF saved as {gif_path}")
