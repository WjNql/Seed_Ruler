import time

from interface.Layout.DL_layout import Ui_Form
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
import os
import sys
import re
from pathlib import Path


class DL_functions(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        print(1)
        # self.ui.pushButton_6.clicked.connect(self.close)

    def open_filepackage(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Folder", "../inference")
        self.ui.lineEdit.setText(directory)

    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", "../inference/images_rice")
        self.ui.lineEdit.setText(file)

    def show_img(self, image_path):
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height = img.shape[0]
            width = img.shape[1]
            ratio = float(height / width)
            new_height = 300
            new_width = int(new_height / ratio)
            img = cv2.resize(img, (new_width, new_height))

            frame = QImage(img, new_width, new_height, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item = QGraphicsPixmapItem(pix)
            self.scene = QGraphicsScene()
            self.scene.addItem(self.item)
            self.ui.graphicsView.setScene(self.scene)
        else:
            QMessageBox.information(self, "Open Failed", "Unable to read the image")

    def start(self):
        path = os.path.split(sys.path[0])[0]
        os.chdir(path)
        temp = self.ui.lineEdit.text()
        result_path = Path(path + "/inference/output")
        path_list = os.listdir(result_path)
        path_list.sort(reverse=True)
        count = 0
        if path_list:
            for filename in path_list:
                filename = re.sub("\D", "", filename)
                filename = int(filename)
                if count < filename:
                    count = filename
            count = count + 1

            print(count)
        end_path = str(result_path) + "/exp" + str(count)
        code = 'python interface/DL_Functions/DL_detect.py --source \"' + temp + '\" --img-size 640 --weights weights/DL_best.pt --conf 0.9 --save-txt'
        print(code)
        os.system(code)
        self.ui.lineEdit_2.setText(end_path)
        images_path = os.path.join(end_path, 'images')
        image_path = os.path.join(images_path, os.listdir(images_path)[0])
        self.show_img(image_path)
        excel_path = os.path.join(end_path, 'DL_rate_data.xlsx')
        if len(excel_path) > 0:
            input_table = pd.read_excel(excel_path)
            input_table_rows = input_table.shape[0]
            input_table_colunms = input_table.shape[1]
            input_table_header = input_table.columns.values.tolist()
            self.ui.tableWidget.setColumnCount(input_table_colunms)
            self.ui.tableWidget.setRowCount(input_table_rows)
            self.ui.tableWidget.setHorizontalHeaderLabels(input_table_header)

            for i in range(input_table_rows):
                input_table_rows_values = input_table.iloc[[i]]
                input_table_rows_values_array = np.array(input_table_rows_values)
                input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                for j in range(input_table_colunms):
                    input_table_items_list = input_table_rows_values_list[j]
                    input_table_items = str(input_table_items_list)
                    newItem = QTableWidgetItem(input_table_items)
                    newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    self.ui.tableWidget.setItem(i, j, newItem)

    def open_picture(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', "Image(*.png *.jpg)")
        if filename:
            img = cv2.imread(str(filename))
            # OpenCV stores images in BGR format, need to convert to RGB for display
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = img.shape[1]
            y = img.shape[0]
            # When displaying images with Qt, convert to QImage first
            self.ui.zoomscale = 1

            QImg = QImage(img, x, y, QImage.Format_RGB888).scaled(1200, 800, Qt.KeepAspectRatio,
                                                                  Qt.SmoothTransformation)
            pix = QPixmap.fromImage(QImg)
            self.item = QGraphicsPixmapItem(pix)
            self.scene = QGraphicsScene()
            self.scene.addItem(self.item)
            self.ui.graphicsView.setScene(self.scene)
            self.ui.graphicsView.show()
        else:
            QMessageBox.information(self, "Open Failed", "No image selected")

    def quit(self):
        self.close()

    def open_data(self):
        # Display corresponding data results
        # Get relevant paths
        openfile_name = QFileDialog.getOpenFileName(self, 'Select File', '../inference/output',
                                                    'Excel files(*.xlsx , *.xls)')
        path_openfile_name = openfile_name[0]
        # Read the table
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

