import sys
from PyQt5.QtWidgets import *
from pandas import read_csv
import ui_predit
class PredIt(QDialog,ui_predit.Ui_PredIt):
    def __init__(self):
        super(PredIt, self).__init__(parent=None)
        self.setupUi(self)
        self.dataset = read_csv("resources/sample.csv")
        self.listofalgotype=['Regression','Classification','Clustering']
        self.Reg_list = ['Simple Linear','Multiple Linear','Polynomial']
        self.Class_list = ['Logistic Regression','K-Nearest Neighbours','SVM'
                           ,'Kernel SVM','Naive Bayes','Decision Tree'
                           'Random Forest']
        self.clust_list = ['K-Means','Hierarchical']
        self.algo_type.addItems(self.listofalgotype)
        self.algo.addItems(self.Reg_list)
        self.algo_type.activated.connect(self.doit)

    def doit(self):
        clicked = self.algo_type.currentIndex()
        self.algo.clear()
        if clicked == 0:
            self.algo.addItems(self.Reg_list)
        elif clicked == 1:
            self.algo.addItems(self.Class_list)
        else:
            self.algo.addItems(self.clust_list)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = PredIt()
    form.show()
    app.exec_()