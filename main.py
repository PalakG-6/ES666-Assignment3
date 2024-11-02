import cv2
import os
from src.Mysubmission.stitcher import PanaromaStitcher


def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def main():
    folders = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']
    print("1 Initializing")
    stitcher = PanaromaStitcher()
    print("2")
    for folder in folders:
        folder_path = f'Images/{folder}'
        print("3")
        print(f"Loading images from {folder_path}...")
        images = load_images_from_folder(folder_path)
        if len(images) < 2:
            print(f"Insufficient images in {folder} to create a panorama. Skipping.")
            continue
        
        # Create panorama
        try:
            panorama, homographies = stitcher.make_panaroma_for_images_in(images)
            print(f"entered for {folder}")
            output_path = f'./results/{folder}_panorama.jpg'
            if not os.path.exists('results'):
                os.makedirs('results')
            cv2.imwrite(output_path, panorama)
            print(f"Panorama created and saved for {folder}!")
        except Exception as e:
            print(f"Failed to create panorama for {folder}. Error: {e}")

if __name__ == "__main__":
    print("hi")
    main()


