from interface.Layout.YOLO_layout import Ui_Form
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
import sys
import re
from interface.SAM_Functions.SAM_RGB_XML import segment_grain
import os


class YOLO_functions(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

    def open_filepackage(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Folder", "../inference")
        self.ui.lineEdit.setText(directory)

    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", "../inference/images_rice")
        self.ui.lineEdit.setText(file)

    def start(self):
        path = os.path.split(sys.path[0])[0]
        os.chdir(path)
        temp = self.ui.lineEdit.text()
        result_path = os.path.join(path, 'inference', 'output')
        path_list = os.listdir(result_path)
        path_list.sort(reverse=True)
        count = 0
        if path_list:
            for filename in path_list:
                filename = re.sub("\\D", "", filename)
                filename = int(filename)
                if count < filename:
                    count = filename
            count = count + 1

            # print(count)
        end_path = result_path + "/exp" + str(count)
        code = 'python interface/YOLO_Functions/YOLO_detect.py --source \"' + temp + '\" --weights weights/YOLO_best.pt --conf 0.3 --agnostic-nms --save-txt'
        print(code)
        os.system(code)
        self.ui.lineEdit_2.setText(end_path)
        path = os.path.split(sys.path[0])[0]
        # Set the working directory to the script's directory
        os.chdir(path)

        # Perform a segmentation process on the specified end_path
        segment_grain(end_path)

        # Display the first image of the current detection round
        file_path = os.path.join(end_path, 'images')
        filename = os.listdir(file_path)[0]
        image_path = os.path.join(file_path, filename)

        if image_path:
            img = cv2.imread(image_path)

            if img is not None:
                # Convert Opencv image from BGR to RGB for display
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x = img.shape[1]
                y = img.shape[0]

                # Get the size of the graphicsView
                view_width = self.ui.graphicsView.width()
                view_height = self.ui.graphicsView.height()

                # Calculate the scaling factors
                scale_x = view_width / x
                scale_y = view_height / y
                scale_factor = min(scale_x, scale_y)

                # Convert to QImage type for Qt image display
                QImg = QImage(img, x, y, QImage.Format_RGB888).scaled(
                    int(x * scale_factor), int(y * scale_factor),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )

                pix = QPixmap.fromImage(QImg)
                self.item = QGraphicsPixmapItem(pix)
                self.scene = QGraphicsScene()
                self.scene.addItem(self.item)
                self.ui.graphicsView.setScene(self.scene)
                self.ui.graphicsView.show()
            else:
                QMessageBox.information(self, "Open Failed", "Unable to read the image")
        else:
            QMessageBox.information(self, "Open Failed", "No image selected")

        # Display the YOLO detection results for the current round
        path_openfile_name = os.path.join(end_path, 'YOLO_rate_data.xlsx')

        # Read the spreadsheet
        if len(path_openfile_name) > 0:
            input_table = pd.read_excel(path_openfile_name)
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
        else:
            QMessageBox.information(self, "Open Failed", "No data file selected")

    def quit(self):
        self.close()
