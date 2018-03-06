import sys
import  os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from pandas import read_csv
import ui_predit
class PredIt(QDialog,ui_predit.Ui_PredIt):
    def __init__(self):
        super(PredIt, self).__init__(parent=None)
        self.setupUi(self)
        self.trainsize =80
        self.testsize = 20
        self.filename = "resources/Data.csv"
        self.trainspinBox.setValue(self.trainsize)
        self.testSpinBox.setMaximum(100)
        self.testSpinBox.setValue(self.testsize)
        self.dataset = read_csv(self.filename,header=None)
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

        self.algo_type.activated.connect(self.setalgotype)
        self.viewpushButton.clicked.connect(self.dispdataset)
        self.openpushButton.clicked.connect(self.open_file)
        self.resultbutton.clicked.connect(self.display_result)

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
        table.setWindowTitle('hERO')
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
            self.dataset = read_csv(self.filename[0], header=None)
            self.finaldataset = self.dataset.iloc[:, :].values
            self.table_row = self.dataset.shape[0]
            self.table_col = self.dataset.shape[1]

    def display_result(self):
        dail = QDialog(self.sender())
        tbrowser = QTextBrowser()
        dail.setWindowTitle('Predit - Result')
        layout = QGridLayout()
        layout.addWidget(tbrowser)
        dail.setLayout(layout)
        tbrowser.setHtml('<html> <head> <!-- your webpage info goes here --> <title>My First Website</title> <meta name="author" content="your name" /> <meta name="description" content="" /> <!-- you should always add your stylesheet (css) in the head tag so that it starts loading before the page html is being displayed --> <link rel="stylesheet" href="style.css" type="text/css" /> </head> <body> <!-- webpage content goes here in the body --> <div id="page"> <div id="logo"> <h1><a href="/" id="logoLink">My First Website</a></h1> </div> <div id="nav"> <ul> <li><a href="#/home.html">Home</a></li> <li><a href="#/about.html">About</a></li> <li><a href="#/contact.html">Contact</a></li> </ul> </div> <div id="content"> <h2>Home</h2> <p> This is my first webpage! I was able to code all the HTML and CSS in order to make it. Watch out world of web design here I come! </p> <p> I can use my skills here to create websites for my business, my friends and family, my C.V, blog or articles. As well as any games or more experiment stuff (which is what the web is really all about). </p> </div> <div id="footer"> <p> Webpage made by <a href="/" target="_blank">[your name]</a> </p> </div> </div> </body> </html>')
        dail.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = PredIt()
    form.show()
    app.exec_()