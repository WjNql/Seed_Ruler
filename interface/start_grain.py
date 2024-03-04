import sys
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from interface.Layout.Main_interface_layout import Ui_MainWindow
import shutil
from PyQt5.QtWidgets import *
from interface.Layout_Fuctions.YOLO_functions import YOLO_functions
from interface.Layout_Fuctions.SAM_functions import SAM_functions
from interface.Layout_Fuctions.DL_functions import DL_functions
from interface.Layout_Fuctions.IP_functions import IP_functions

class MyHomePage(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyHomePage, self).__init__()
        self.setupUi(self)

        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.camera.set(cv2.CAP_PROP_FPS, 60)
        self.is_camera_opened = False  # Indicates whether the camera is open or not
        # Timer to capture a frame every 30ms
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self.startfilename = "startgrain.png"

    # Open the camera
    def open_camera_click(self):
        """
        Open and close the camera
        """
        self.is_camera_opened = ~self.is_camera_opened
        if self.is_camera_opened:
            self.button2.setText("Close Camera")
            self._timer.start()
        else:
            self.button2.setText("Open Camera")
            self._timer.stop()

    # Function to capture an image from the video
    def camera_click(self):
        if not self.is_camera_opened:
            return
        # Capture the video
        self.button2.setText("Open Camera")
        self._timer.stop()
        self.is_camera_opened = ~self.is_camera_opened
        temp = self.src
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.startfilename, temp)
        img_rows, img_cols, channels = self.src.shape
        bytePerLine = channels * img_cols
        # Convert to QImage for displaying in Qt
        QImg = QImage(self.src.data, img_cols, img_rows, bytePerLine, QImage.Format_RGB888)
        self.label_3.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_3.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    # Open a file and display the original image
    def openfile_click(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', '../inference/images_rice', "Image(*.png *jpg)")
        # If an existing file is selected, use its path directly
        if filename:
            self.src = cv2.imread(str(filename))  # Original image, to be used in many places, hence 'self'
            # OpenCV stores images in BGR format, so conversion from BGR to RGB is required for display
            shutil.copyfile(filename, self.startfilename)
            temp = cv2.cvtColor(self.src, cv2.COLOR_BGR2RGB)
            rows, cols, channels = temp.shape
            bytePerLine = channels * cols
            QImg = QImage(temp.data, cols, rows, bytePerLine, QImage.Format_RGB888)
            self.label_3.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.label_3.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            QMessageBox.information(self, "Open Failed", "No image selected")

    # Run algorithms
    def function1_click(self):
        # Germination rate measurement
        print('aa')
        self.child1 = YOLO_functions()
        self.child1.show()

    def function2_click(self):
        print('bb')
        self.child2 = SAM_functions()
        self.child2.show()

    def function3_click(self):
        print('cc')
        self.child4 = DL_functions()
        self.child4.show()

    def function4_click(self):
        print('dd')
        self.child5 = IP_functions()
        self.child5.show()

    # Save results
    def save_click(self):
        savefilename, _ = QFileDialog.getSaveFileName(self, "Save Image", '..\inference\images_rice', 'Image(*.png;*.jpg)')
        if savefilename:
            shutil.copyfile(self.startfilename, savefilename)
            QMessageBox.information(self, "Save Successful", "Save successful")
        else:
            QMessageBox.information(self, "Save Failed", "Please choose a valid path")
            return

    def _queryFrame(self):
        """
        Capture frames continuously
        """
        ret, self.src = self.camera.read()
        img_rows, img_cols, channels = self.src.shape
        bytePerLine = channels * img_cols
        self.src = cv2.flip(self.src, 1)
        self.src = cv2.cvtColor(self.src, cv2.COLOR_BGR2RGB)
        QImg = QImage(self.src.data, img_cols, img_rows, bytePerLine, QImage.Format_RGB888)
        self.label_3.setPixmap(QPixmap.fromImage(QImg).scaled(self.label_3.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def quit(self):
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ex = MyHomePage()
    ex.show()
    sys.exit(app.exec_())
