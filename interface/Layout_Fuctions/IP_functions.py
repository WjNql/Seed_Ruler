from interface.Layout.IP_layout import Ui_Form
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
import os
import re
from interface.IP_Functions.test_d import getRateofseeds


class IP_functions(QWidget):
    def __init__(self,parent = None):
        super().__init__(parent)
        self.ui=Ui_Form()
        self.ui.setupUi(self)


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
            new_height = 500
            new_width = int(new_height / ratio)
            img = cv2.resize(img, (new_width, new_height))

            frame = QImage(img, new_width, new_height, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item = QGraphicsPixmapItem(pix)
            self.scene = QGraphicsScene()  # 创建场景
            self.scene.addItem(self.item)
            self.ui.graphicsView.setScene(self.scene)
        else:
            QMessageBox.information(self, "打开失败", "无法读取图片")

    def start(self):
        img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
        imgOrdir_path = self.ui.lineEdit.text()
        print(os.getcwd())
        result_path = "../inference/output"
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
        save_path = result_path + "/exp" + str(count)
        os.makedirs(save_path, exist_ok=True)
        df_list = []
        if os.path.isfile(imgOrdir_path):
            df_list.append(getRateofseeds(imgOrdir_path, save_path))
            image_path = os.path.join(save_path, os.listdir(save_path)[0])
            self.show_img(image_path)
        elif os.path.isdir(imgOrdir_path):
            for file in os.listdir(imgOrdir_path):
                if file.split('.')[-1].lower() in img_formats:
                    save_dir = os.path.join(save_path, file.split('.')[0])  # 为每张图片创建文件夹
                    os.makedirs(save_dir, exist_ok=True)
                    df_list.append(getRateofseeds(os.path.join(imgOrdir_path, file), save_dir))
            dir_path = os.path.join(save_path, os.listdir(save_path)[0])
            image_path = os.path.join(dir_path, os.listdir(dir_path)[0])
            self.show_img(image_path)
        # 使用pandas.concat将DataFrame的列表合并为一个DataFrame
        df = pd.DataFrame(df_list)
        # 保存DataFrame到Excel文件
        excel_path = os.path.join(save_path, "IP_rate_data.xlsx")
        df.to_excel(excel_path, index=False)
        self.ui.lineEdit_2.setText(os.path.abspath(save_path))

        # 读取表格
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
        else:
            QMessageBox.information(self, "打开失败", "没有选择数据文件")



    def quit(self):
        self.close()
