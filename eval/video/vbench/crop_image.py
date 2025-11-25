import os
from PIL import Image

def center_crop_images(input_dir, output_dir, aspect_ratio="16:9"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
    target_ratio = w_ratio / h_ratio

    for file_name in os.listdir(input_dir):
        if not file_name.lower().endswith(".jpg"):
            continue

        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        with Image.open(input_path) as img:
            width, height = img.size
            current_ratio = width / height

            if current_ratio > target_ratio:
                new_width = int(height * target_ratio)
                new_height = height
            else:
                new_width = width
                new_height = int(width / target_ratio)

            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height

            cropped = img.crop((left, top, right, bottom))
            cropped.save(output_path, "JPEG")

    print(f"save images to: {output_dir}")


if __name__ == "__main__":
    aspect_ratio="16:9"
    input_dir = "./vbench/origin"
    output_dir = "./vbench/origin_crop"
    center_crop_images(input_dir, output_dir, aspect_ratio)