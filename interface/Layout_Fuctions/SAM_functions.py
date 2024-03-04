from interface.Layout.SAM_layout import Ui_Form
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
import shutil
import os
from .Click_functions import Click_functions


def clear_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        print(f"Successfully cleared folder: {folder_path}")
    except Exception as e:
        print(f"Error clearing folder {folder_path}: {e}")


class SAM_functions(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.scene = None

    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", "../inference/output")
        self.ui.lineEdit.setText(file)
        self.image_path = file

        if file:
            img = cv2.imread(str(file))
            if img is not None:
                # OpenCV stores images in BGR format, need to convert to RGB for display
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x = img.shape[1]
                y = img.shape[0]

                # Calculate the scaling ratio while maintaining the aspect ratio of the image
                view_width = self.ui.graphicsView.width()
                view_height = self.ui.graphicsView.height()
                scale_ratio = min(view_width / x, view_height / y)

                # Calculate the offset to center the image
                offset_x = (view_width - x * scale_ratio) / 2
                offset_y = (view_height - y * scale_ratio) / 2

                # When displaying images with Qt, convert to QImage first
                self.ui.zoomscale = scale_ratio

                QImg = QImage(img.data, x, y, img.strides[0], QImage.Format_RGB888)
                QImg = QImg.scaled(x * scale_ratio, y * scale_ratio, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                pix = QPixmap.fromImage(QImg)

                # If the scene already exists, clear the scene; otherwise, create a new scene
                if self.scene is not None:
                    self.scene.clear()
                else:
                    self.scene = QGraphicsScene()

                self.item = QGraphicsPixmapItem(pix)
                self.item.setPos(offset_x, offset_y)  # Set the offset position
                self.scene.addItem(self.item)
                self.ui.graphicsView.setScene(self.scene)
                self.ui.graphicsView.show()
            else:
                QMessageBox.information(self, "Open Failed", "Unable to read image file")
        else:
            QMessageBox.information(self, "Open Failed", "No image selected")

    def start(self):
        filename = self.image_path.split('/')[-1].split('.')[0]
        single_excel_path = os.path.join(os.path.dirname(os.path.dirname(self.image_path)), 'result_data.xlsx')
        all_excel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.image_path))),
                                      'result_data.xlsx')
        single_df = pd.read_excel(single_excel_path)
        all_df = pd.read_excel(all_excel_path)
        area_value = single_df.loc[single_df["Name"] == filename, "Area"].values[
            0]  # Read the area of the selected image
        origin_df = all_df["Name"]  # Get all grain names for later calculation of the number of grains in the image
        loc_origin_df = all_df[
            ["Name", "TopLeft_x", "TopLeft_y", "BottomRight_x", "BottomRight_y"]]  # Original coordinates of all grains
        loc_after_df = all_df.loc[all_df["Area"] >= area_value, ["Name", "TopLeft_x", "TopLeft_y", "BottomRight_x",
                                                                 "BottomRight_y"]]  # Get coordinates of grains after selection
        # Columns "Name", "TopLeft_x", "TopLeft_y", "BottomRight_x", "BottomRight_y" uniquely identify a grain
        unique_columns = ["Name", "TopLeft_x", "TopLeft_y", "BottomRight_x", "BottomRight_y"]
        # Get the intersection of the two DataFrames
        intersection_df = pd.merge(loc_after_df, loc_origin_df, on=unique_columns, how="inner")
        # Get the difference between loc_after_df and loc_origin_df
        difference_df = loc_origin_df.loc[
            ~loc_origin_df.set_index(unique_columns).index.isin(intersection_df.set_index(unique_columns).index)]
        inter_name_list = list(set(intersection_df["Name"].tolist()))  # Get unique file names in the intersection
        dif_name_list = list(set(difference_df["Name"].tolist()))  # Get unique file names in the difference
        inter_list = intersection_df.to_dict(
            orient='records')  # Dictionary with keys as "Name" and values as corresponding row values (intersection)
        dif_list = difference_df.to_dict(
            orient='records')  # Dictionary with keys as "Name" and values as corresponding row values (difference)
        sam_label_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.image_path)))),
            'SAM_labeled_images')  # Location of SAM segmented labeled images
        os.makedirs(sam_label_dir, exist_ok=True)

        # Label YES
        for count, name in enumerate(inter_name_list):
            filtered_values = [d for d in inter_list if
                               d['Name'] == name]  # Get the dictionary of germinated grains in the current image name
            result_list = [list(d.values())[1:] for d in
                           filtered_values]  # Get the coordinates of germinated grains in the current image name
            sam_img_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.image_path)))),
                'SAM_images', name + '.png')  # SAM segmented image filename
            sam_label_path = os.path.join(sam_label_dir, name + '.png')  # SAM segmented labeled image name
            sam_img = cv2.imread(sam_img_path)
            # Set rectangle color (B, G, R)
            color = (255, 0, 0)
            # Set line thickness
            thickness = 5
            for value in result_list:
                x1, y1, x2, y2 = value[0], value[1], value[2], value[3]
                # Use cv2.rectangle to draw a rectangle
                cv2.rectangle(sam_img, (x1, y1), (x2, y2), color, thickness)
                # Add text above the rectangle
                text = "Yes"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                text_color = (255, 255, 255)
                text_thickness = 1
                text_position = (x1, y1 - 10)  # Set text position above the rectangle
                cv2.putText(sam_img, text, text_position, font, font_scale, text_color, text_thickness)
            cv2.imwrite(sam_label_path, sam_img)

        # Label NO
        for count, name in enumerate(dif_name_list):
            filtered_values = [d for d in dif_list if d[
                'Name'] == name]  # Get the dictionary of non-germinated grains in the current image name
            result_list = [list(d.values())[1:] for d in
                           filtered_values]  # Get the coordinates of non-germinated grains in the current image name
            sam_label_path = os.path.join(sam_label_dir, name + '.png')  # SAM segmented labeled image name
            sam_img = cv2.imread(sam_label_path)
            # Set rectangle color (B, G, R)
            color = (0, 0, 255)
            # Set line thickness
            thickness = 5
            for value in result_list:
                x1, y1, x2, y2 = value[0], value[1], value[2], value[3]
                # Use cv2.rectangle to draw a rectangle
                cv2.rectangle(sam_img, (x1, y1), (x2, y2), color, thickness)
                # Add text above the rectangle
                text = "No"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                text_color = (255, 255, 255)
                text_thickness = 1
                text_position = (x1, y1 - 10)  # Set text position above the rectangle
                cv2.putText(sam_img, text, text_position, font, font_scale, text_color, text_thickness)
            cv2.imwrite(sam_label_path, sam_img)

        all_df = all_df.loc[
            all_df["Area"] >= area_value, ["Name"]]  # Get all names with an area greater than the selected image area
        origin_df_list = origin_df.values.tolist()
        all_df_list = all_df.values.tolist()
        all_df_list = [str(x) for item in all_df_list for x in item]  # Remove square brackets from the list elements
        origin_count = {}
        all_count = {}
        for i in origin_df_list:
            origin_count[i] = origin_df_list.count(
                i)  # Calculate the number of grains in each image in the original data
        for i in all_df_list:
            all_count[i] = all_df_list.count(
                i)  # Calculate the remaining number of grains in each image after selection
        for key in origin_count:  # Prevent a situation where an image has no remaining grains
            if key not in all_count.keys():
                all_count[key] = 0
        result_excel = pd.DataFrame()
        name_index = list(origin_count.keys())
        yes_number = list(all_count.values())
        total_number = list(origin_count.values())
        no_number = list(int(total_number[i]) - int(yes_number[i]) for i in range(len(total_number)))
        rate = list(round(int(yes_number[i]) / int(total_number[i]), 4) for i in range(len(total_number)))
        result_excel["name_index"] = name_index
        result_excel["yes_number"] = yes_number
        result_excel["no_number"] = no_number
        result_excel["total_number"] = total_number
        result_excel["rate"] = rate
        excel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.image_path)))),
                                  'SAM_rate_data.xlsx')
        result_excel.to_excel(excel_path)
        print(f"Data saved to {excel_path}")
        self.ui.lineEdit_2.setText(excel_path)

        # Display the Excel file
        # Read the table
        if len(excel_path) > 0:
            input_table = pd.read_excel(excel_path)
            input_table_rows = input_table.shape[0]
            input_table_columns = input_table.shape[1]
            input_table_header = input_table.columns.values.tolist()
            self.ui.tableWidget.setColumnCount(input_table_columns)
            self.ui.tableWidget.setRowCount(input_table_rows)
            self.ui.tableWidget.setHorizontalHeaderLabels(input_table_header)

            for i in range(input_table_rows):
                input_table_rows_values = input_table.iloc[[i]]
                input_table_rows_values_array = np.array(input_table_rows_values)
                input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                for j in range(input_table_columns):
                    input_table_items_list = input_table_rows_values_list[j]
                    input_table_items = str(input_table_items_list)
                    newItem = QTableWidgetItem(input_table_items)
                    newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    self.ui.tableWidget.setItem(i, j, newItem)

    def quit(self):
        self.close()

    def choose_picture(self):
        file, _ = QFileDialog.getOpenFileName(self, "选取文件", "../inference/output")
        if file:
            self.ui.lineEdit.setText(file)
            self.child1 = Click_functions(self.ui.lineEdit.text())
            self.child1.mouse_clicked.connect(self.handle_child_click)
            self.child1.show()
        else:
            QMessageBox.information(self, "打开失败", "没有选择图片")

    def handle_child_click(self, pos):
        self.child1.close()
        filename = self.ui.lineEdit.text().split('/')[-1].split('.')[0]
        path_grain = os.path.join(os.path.dirname(os.path.dirname(self.ui.lineEdit.text())), 'grain_information',
                                  filename)
        path_openfile_name = os.path.join(path_grain, 'result_data.xlsx')
        df = pd.read_excel(path_openfile_name)
        name_list = df["Name"].tolist()
        lefttop_x = df["TopLeft_x"].tolist()
        lefttop_y = df["TopLeft_y"].tolist()
        rightbot_x = df["BottomRight_x"].tolist()
        rightbot_y = df["BottomRight_y"].tolist()
        for i in range(len(lefttop_x)):
            if lefttop_x[i] < pos.x() < rightbot_x[i] and lefttop_y[i] < pos.y() < rightbot_y[i]:
                image_name = name_list[i]
                self.image_path = os.path.join(path_grain, image_name, image_name + '.png')
                img = cv2.imread(self.image_path)
                if img is not None:
                    # OpenCV stores images in BGR format, need to convert to RGB for display
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    x = img.shape[1]
                    y = img.shape[0]

                    # Calculate the scaling ratio while maintaining the aspect ratio of the image
                    view_width = self.ui.graphicsView.width()
                    view_height = self.ui.graphicsView.height()
                    scale_ratio = min(view_width / x, view_height / y)

                    # Calculate the offset to center the image
                    offset_x = (view_width - x * scale_ratio) / 2
                    offset_y = (view_height - y * scale_ratio) / 2

                    # When displaying images with Qt, convert to QImage first
                    self.ui.zoomscale = scale_ratio

                    QImg = QImage(img.data, x, y, img.strides[0], QImage.Format_RGB888)
                    QImg = QImg.scaled(x * scale_ratio, y * scale_ratio, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                    pix = QPixmap.fromImage(QImg)

                    # If the scene already exists, clear the scene; otherwise, create a new scene
                    if self.scene is not None:
                        self.scene.clear()
                    else:
                        self.scene = QGraphicsScene()

                    self.item = QGraphicsPixmapItem(pix)
                    self.item.setPos(offset_x, offset_y)  # Set the offset position
                    self.scene.addItem(self.item)
                    self.ui.graphicsView.setScene(self.scene)
                    self.ui.graphicsView.show()
                else:
                    QMessageBox.information(self, "Open Failed", "Unable to read image file")

