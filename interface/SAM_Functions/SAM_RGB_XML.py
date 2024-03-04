import os
import cv2
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ET
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = (mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255).astype(np.uint8)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR format
    return mask_image


# RGB值计算
def rgb_calculate(image, filenum, save_path):
    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Calculate |R-B| + |G-B|
    result_array = (np.abs(image_array[:, :, 0] - image_array[:, :, 2]) +
                    np.abs(image_array[:, :, 1] - image_array[:, :, 2]))

    # Set the threshold
    threshold = 5  # Adjust the threshold as needed

    # Threshold segmentation
    result_array[result_array >= threshold] = 0

    # Create an image to display the calculation result
    binary_image = Image.fromarray(result_array.astype(np.uint8))
    binary_array = np.array(binary_image)

    # Dilate operation to fill holes in the embryo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_image = cv2.dilate(binary_array, kernel)

    # Find contours
    try:
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)

        # Get the rectangle of the contour
        rect = cv2.boundingRect(max_contour)

        # Get the embryo area
        embryo_image = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        embryo_area = dilated_image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

        # Save the embryo area image
        embryo_img = filenum + "_embryo.png"
        output_path = os.path.join(save_path, embryo_img)
        cv2.imwrite(output_path, embryo_image)

        # Calculate the perimeter and area of the embryo region
        perimeter = cv2.arcLength(max_contour, True)
        area = cv2.contourArea(max_contour)
        pixel = cv2.countNonZero(embryo_area)

        txt_name = filenum + ".txt"
        txt_path = os.path.join(save_path, txt_name)

        # Use the open function to create or open a file, 'w' indicates write mode
        with open(txt_path, 'w') as file:
            # Write content to the file
            file.write(f"Perimeter: {perimeter}\n")
            file.write(f"Area: {area}\n")
            file.write(f"Pixel Count: {pixel}\n")

        return area
    except ValueError as e:
        print(f"Error processing {filenum}: {e}")
        return 0


# Read annotation information from XML file

def read_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        obj_info = {}
        obj_info['name'] = obj.find('name').text
        bndbox = obj.find('bndbox')
        obj_info['xmin'] = int(bndbox.find('xmin').text)
        obj_info['ymin'] = int(bndbox.find('ymin').text)
        obj_info['xmax'] = int(bndbox.find('xmax').text)
        obj_info['ymax'] = int(bndbox.find('ymax').text)
        objects.append(obj_info)

    return filename, width, height, objects


# 精细分割
def segment_grain(path):
    # SAM preparation
    sys.path.append("../..")

    # Use a small model
    sam_checkpoint = "./weights/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Set image parameters
    img_total = []
    output_dir = os.path.join(path, "grain_information")  # Save directory
    os.makedirs(output_dir, exist_ok=True)
    images_xml_dir = os.path.join(path, "images_xml")
    all_df_list = []  # Save all grain data in all images
    # plt.figure(figsize=(10, 10))  # Create a graphics window
    for file in os.listdir(images_xml_dir):
        if file.endswith(".xml"):
            img_total.append(file.split('.')[0])

    for xml_name in img_total:
        xml_file = os.path.join(images_xml_dir, f"{xml_name}.xml")
        filename, width, height, objects = read_xml(xml_file)
        df_list = []  # Save all grain data in a single image
        image = cv2.imread(os.path.join(images_xml_dir, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        folder_path = os.path.join(output_dir, xml_name)  # Create a folder for each image
        os.makedirs(folder_path, exist_ok=True)

        box = []
        for obj_info in objects:
            box.append([obj_info['xmin'], obj_info['ymin'], obj_info['xmax'], obj_info['ymax']])
            input_boxes = torch.tensor(box, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        for i, mask in enumerate(masks):
            mask = ~mask
            mask = mask + 255
            mask = np.repeat(mask.cpu()[0].numpy()[:, :, np.newaxis], 3, axis=2)
            mask = mask.astype(np.uint8)
            res = cv2.bitwise_and(image, mask)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            res = res[box[i][1]:box[i][3], box[i][0]:box[i][2]]
            filenum = xml_name + '_' + str(i + 1)  # Create a folder for each grain
            save_path = os.path.join(folder_path, filenum)
            os.makedirs(save_path, exist_ok=True)
            filename = filenum + '.png'
            cv2.imwrite(os.path.join(save_path, filename), res)  # Save segmented images
            area = rgb_calculate(res, filenum, save_path)
            df_list.append({
                "Name": filenum,
                "Area": area,
                "TopLeft_x": box[i][0],
                "TopLeft_y": box[i][1],
                "BottomRight_x": box[i][2],
                "BottomRight_y": box[i][3],
            })
            all_df_list.append({
                "Name": xml_name,
                "Area": area,
                "TopLeft_x": box[i][0],
                "TopLeft_y": box[i][1],
                "BottomRight_x": box[i][2],
                "BottomRight_y": box[i][3],
            })
            # Use pandas.concat to merge the list of DataFrames into one DataFrame
            df = pd.DataFrame(df_list)

            # Save DataFrame to an Excel file
            excel_path = os.path.join(folder_path, "result_data.xlsx")
            df.to_excel(excel_path, index=False)

        # # Clear previous graphics
        # plt.clf()

        # Get the base mask
        base_mask = masks.cpu().numpy()[0][0]

        # Add up the masks
        for mask in masks.cpu().numpy()[1:]:
            for i in range(len(base_mask)):
                base_mask[i] = np.add(base_mask[i], mask[0][i])

        # Show the mask image
        mask_image = show_mask(base_mask, random_color=False)

        # Save the image
        sam_path = os.path.join(path, "SAM_images")
        os.makedirs(sam_path, exist_ok=True)
        cv2.imwrite(os.path.join(sam_path, xml_name + '.png'), mask_image)
        print(f"SAM_image saved to {sam_path}")

        # Use pandas.concat to merge the list of DataFrames into one DataFrame
        all_df = pd.DataFrame(all_df_list)

        # Save DataFrame to an Excel file
        excel_path = os.path.join(output_dir, "result_data.xlsx")
        all_df.to_excel(excel_path, index=False)


if __name__ == '__main__':
    image_path = "../inference/output/exp2"
    segment_grain(image_path)
