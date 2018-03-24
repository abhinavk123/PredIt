import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from malgos import *
from pandas import read_csv
import ui_predit
import resources
class PredIt(QDialog,ui_predit.Ui_PredIt):
    def __init__(self):
        super(PredIt, self).__init__(parent=None)
        self.setupUi(self)
        self.setWindowIcon(QIcon(":/icon"))
        self.trainsize =80
        self.testsize = 20
        self.filename = "resources/Data.csv"
        self.trainspinBox.setValue(self.trainsize)
        self.testSpinBox.setMaximum(100)
        self.testSpinBox.setValue(self.testsize)
        self.dataset = read_csv(self.filename)
        self.finaldataset  = self.dataset.iloc[:,:].values
        self.table_row = self.dataset.shape[0]
        self.table_col = self.dataset.shape[1]

        self.listofalgotype=['Regression','Classification','Clustering']
        self.Reg_list = ['Simple Linear','Multiple Linear','Polynomial']
        self.Class_list = ['Logistic Regression','K-Nearest Neighbours','SVM'
                           ,'Kernel SVM','Naive Bayes','Decision Tree',
                           'Random Forest']
        self.clust_list = ['K-Means','Hierarchical']

        self.algo_type.addItems(self.listofalgotype)
        self.algo.addItems(self.Reg_list)

        self.algo_type.activated.connect(self.setalgotype)
        self.viewpushButton.clicked.connect(self.dispdataset)
        self.openpushButton.clicked.connect(self.open_file)
        self.resultbutton.clicked.connect(self.compute_result)

    def setalgotype(self):
        clicked = self.algo_type.currentIndex()
        self.algo.clear()
        if clicked == 0:
            self.algo.addItems(self.Reg_list)
        elif clicked == 1:
            self.algo.addItems(self.Class_list)
        else:
            self.algo.addItems(self.clust_list)

    def dispdataset(self):
        wid = QDialog(self.sender())
        table = QTableWidget()
        table.setWindowTitle('PredIt - Result')
        table.setRowCount(self.table_row)
        table.setColumnCount(self.table_col)
        for i in range(self.table_row):
            for j in range(self.table_col):
                itemm = QTableWidgetItem(str(self.finaldataset[i][j]),)
                table.setItem(i,j,itemm)
        table.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        table.sizeAdjustPolicy()
        size = table.size()
        grid = QGridLayout()
        grid.addWidget(table)
        wid.setLayout(grid)
        wid.setWindowTitle(os.path.basename(self.filename[0]))
        wid.resize(size)
        wid.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        wid.exec_()

    @pyqtSlot(int)
    def on_trainspinBox_valueChanged(self,v):
        self.testSpinBox.setValue(100-v)

    def open_file(self):
        dir = "."
        formats = ["*.csv"]
        fname = QFileDialog.getOpenFileName(self, "PredIt - Choose Dataset", dir,
                                            "CSV File (%s)" % " ".join(formats))
        if len(fname[0]):
            self.filename = fname
            self.viewpushButton.setText(os.path.basename(self.filename[0]))
            self.dataset = read_csv(self.filename[0])
            self.finaldataset = self.dataset.iloc[:, :].values
            self.table_row = self.dataset.shape[0]
            self.table_col = self.dataset.shape[1]

    def display_result(self,result):
        dail = QDialog(self.sender())
        tbrowser = QTextBrowser()
        dail.setWindowTitle('Predit - Result')
        layout = QGridLayout()
        layout.addWidget(tbrowser)
        dail.setLayout(layout)
        tbrowser.setText(result)
        tbrowser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tbrowser.sizeAdjustPolicy()
        size = tbrowser.size()
        dail.resize(size)
        dail.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        dail.exec_()


    def compute_result(self):
        type = self.algo_type.currentIndex()
        alg = self.algo.currentIndex()
        if type == 0 :
            if alg == 0:
                self.temp = LinearReg(self.dataset)
                self.display_result(self.temp.disp_result())
            if alg == 1:
                self.temp = MultipleReg(self.dataset)
                self.display_result(self.temp.disp_result())
        if type == 1:
            if alg == 0:
                self.temp = Logs_Reg(self.dataset)
                self.display_result(self.temp.disp_result())
            if alg == 1:
                self.temp = Knn(self.dataset)
                self.display_result(self.temp.disp_result())
            if alg == 2:
                self.temp = SVM(self.dataset)
                self.display_result(self.temp.disp_result())
            if alg == 3:
                self.temp = KernelSvm(self.dataset)
                self.display_result(self.temp.disp_result())
            if alg == 4:
                self.temp = NaiveBayse(self.dataset)
                self.display_result(self.temp.disp_result())
            if alg == 5:
                self.temp = DecisionTree(self.dataset)
                self.display_result(self.temp.disp_result())
            if alg == 6:
                self.temp = RandomForest(self.dataset)
                self.display_result(self.temp.disp_result())

        if type == 2:
            if alg ==0:
                self.temp = KMeans(self.dataset)
                self.display_result(self.temp.disp_result())
            if alg == 1:
                self.temp = HC(self.dataset)
                self.display_result(self.temp.disp_result())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = PredIt()
    form.show()
    app.exec_()