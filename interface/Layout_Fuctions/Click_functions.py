import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from interface.Layout.Main_interface_layout import Ui_MainWindow


class Click_functions(QMainWindow, Ui_MainWindow):
    # Define a signal to transmit the clicked coordinates
    mouse_clicked = pyqtSignal(QPoint)

    def __init__(self, image_path):
        super(Click_functions, self).__init__()
        self.setWindowTitle("Image Window")

        # Create a label for displaying the image
        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        # Read the image
        self.pixmap = QPixmap(image_path)
        self.label.setPixmap(self.pixmap)

        # Set the label click event
        self.label.mousePressEvent = self.mousePressEvent

    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            pos = event.pos()
            x, y = pos.x(), pos.y()
            self.mouse_clicked.emit(pos)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Image path
    path = "../inference/output/exp0/images/N2-7-49B.jpg"
    # Create the window
    window = MyShow3(path)
    window.show()
    sys.exit(app.exec())
