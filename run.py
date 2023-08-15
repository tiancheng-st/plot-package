import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QMessageBox, QLabel, QDialog
from PyQt5.QtCore import QUrl, QSize
from PyQt5.QtGui import QMovie
from PyQt5 import uic
import beamProfile
import crossSection
import percentMap

class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        uic.loadUi('mainWindow.ui', self)
        self.setWindowTitle('Plot')
        self.browse.clicked.connect(self.showFileBrowser)
        self.path = None
        self.beam.clicked.connect(self.runBeamProfile)
        self.cross.clicked.connect(self.runCrossSection)
        self.percent.clicked.connect(self.runPercentageMap)

    def showFileBrowser(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose File to Run", "", "Text Files (*.txt)", options=options)
        if file_path:
            print('File Path:', file_path)
            self.path = file_path
            self.filepath.setText(self.path)


    def runBeamProfile(self):
        beamProfile.convertRadiantToPolarMapAndPlot(self.path)
    
    def runCrossSection(self):
        crossSection.plot_cross_section(self.path)

    def runPercentageMap(self):
        percentMap.run_percentageMap(self.path)


class message(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi('msg.ui', self)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Example()
    window.show()

    app.exec()