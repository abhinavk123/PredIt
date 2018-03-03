import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from pandas import read_csv
import ui_predit
class PredIt(QDialog,ui_predit.Ui_PredIt):
    def __init__(self):
        super(PredIt, self).__init__(parent=None)
        self.setupUi(self)
        self.trainsize =0
        self.testsize = 100

        self.trainspinBox.setValue(self.trainsize)
        self.testSpinBox.setMaximum(100)
        self.testSpinBox.setValue(self.testsize)
        self.dataset = read_csv("resources/Data.csv",header=None)
        self.finaldataset  = self.dataset.iloc[:,:].values
        self.table_row = self.dataset.shape[0]
        self.table_col = self.dataset.shape[1]

        self.listofalgotype=['Regression','Classification','Clustering']
        self.Reg_list = ['Simple Linear','Multiple Linear','Polynomial']
        self.Class_list = ['Logistic Regression','K-Nearest Neighbours','SVM'
                           ,'Kernel SVM','Naive Bayes','Decision Tree'
                           'Random Forest']
        self.clust_list = ['K-Means','Hierarchical']

        self.algo_type.addItems(self.listofalgotype)
        self.algo.addItems(self.Reg_list)

        self.algo_type.activated.connect(self.doit)
        self.viewpushButton.clicked.connect(self.doe)

    def doit(self):
        clicked = self.algo_type.currentIndex()
        self.algo.clear()
        if clicked == 0:
            self.algo.addItems(self.Reg_list)
        elif clicked == 1:
            self.algo.addItems(self.Class_list)
        else:
            self.algo.addItems(self.clust_list)

    def doe(self):
        wid = QDialog()
        table = QTableWidget()
        table.setWindowTitle('hERO')
        table.setRowCount(self.table_row)
        table.setColumnCount(self.table_col)
        for i in range(self.table_row):
            for j in range(self.table_col):
                itemm = QTableWidgetItem(str(self.finaldataset[i][j]),)
                table.setItem(i,j,itemm)
        grid = QGridLayout()
        grid.addWidget(table,0,0)
        wid.setLayout(grid)
        wid.setWindowTitle('Hello')
        wid.exec_()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = PredIt()
    form.show()
    app.exec_()