from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

class LinearReg():
    def __init__(self,dataset):
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        self.msg = ""
        for i in range(len(y_pred)):
            self.msg += "Predicated Value :" + str(y_pred[i]) + "     Expected Value  :" +str(y_test[i]) +"\n"

        #display_result(msg)

    def disp_result(self):
        return self.msg




class MultipleReg():
    def __init__(self,dataset):
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 4].values
        labelencoder = LabelEncoder()
        X[:, 3] = labelencoder.fit_transform(X[:, 3])
        onehotencoder = OneHotEncoder(categorical_features=[3])
        X = onehotencoder.fit_transform(X).toarray()
        X = X[:, 1:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        self.msg = ""
        for i in range(len(y_pred)):
            self.msg += "Predicated Value :" + str(y_pred[i]) + "     Expected Value  :" + str(y_test[i]) + "\n"

    def disp_result(self):
        return self.msg

class PolyReg():
    def __init__(self,dataset):
        X = dataset.iloc[:, 1:2].values
        y = dataset.iloc[:, 2].values
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        poly_reg = PolynomialFeatures(degree=4)
        X_poly = poly_reg.fit_transform(X)
        poly_reg.fit(X_poly, y)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(X_poly, y)


class Logs_Reg():
    def __init__(self,dataset):

        X = dataset.iloc[:, [2, 3]].values
        y = dataset.iloc[:, 4].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        self.msg = ""
        for i in range(len(y_pred)):
            self.msg += "Predicated Value :" + str(y_pred[i]) + "     Expected Value  :" + str(y_test[i]) + "\n"

    def disp_result(self):
        return self.msg

class Knn():
    def __init__(self,dataset):
        X = dataset.iloc[:, [2, 3]].values
        y = dataset.iloc[:, 4].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        self.msg = ""
        for i in range(len(y_pred)):
            self.msg += "Predicated Value :" + str(y_pred[i]) + "     Expected Value  :" + str(y_test[i]) + "\n"

    def disp_result(self):
        return self.msg

class SVM():
    def __init__(self,dataset):
        X = dataset.iloc[:, [2, 3]].values
        y = dataset.iloc[:, 4].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        from sklearn.svm import SVC
        classifier = SVC(kernel='linear', random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        self.msg = ""
        for i in range(len(y_pred)):
            self.msg += "Predicated Value :" + str(y_pred[i]) + "     Expected Value  :" + str(y_test[i]) + "\n"

    def disp_result(self):
        return self.msg

class KernelSvm():
    def __init__(self,dataset):
        X = dataset.iloc[:, [2, 3]].values
        y = dataset.iloc[:, 4].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        from sklearn.svm import SVC
        classifier = SVC(kernel='rbf', random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        self.msg = ""
        for i in range(len(y_pred)):
            self.msg += "Predicated Value :" + str(y_pred[i]) + "     Expected Value  :" + str(y_test[i]) + "\n"

    def disp_result(self):
        return self.msg

class NaiveBayse():
    def __init__(self,dataset):
        X = dataset.iloc[:, [2, 3]].values
        y = dataset.iloc[:, 4].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        self.msg = ""
        for i in range(len(y_pred)):
            self.msg += "Predicated Value :" + str(y_pred[i]) + "     Expected Value  :" + str(y_test[i]) + "\n"

    def disp_result(self):
        return self.msg

class DecisionTree():
    def __init__(self,dataset):
        X = dataset.iloc[:, [2, 3]].values
        y = dataset.iloc[:, 4].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        self.msg = ""
        for i in range(len(y_pred)):
            self.msg += "Predicated Value :" + str(y_pred[i]) + "     Expected Value  :" + str(y_test[i]) + "\n"

    def disp_result(self):
        return self.msg

class RandomForest():
    def __init__(self,dataset):
        X = dataset.iloc[:, [2, 3]].values
        y = dataset.iloc[:, 4].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        self.msg = ""
        for i in range(len(y_pred)):
            self.msg += "Predicated Value :" + str(y_pred[i]) + "     Expected Value  :" + str(y_test[i]) + "\n"

    def disp_result(self):
        return self.msg

class KMeans():
    def __init__(self,dataset):
        X = dataset.iloc[:, [3, 4]].values
        y = dataset.iloc[:, 3].values
        from sklearn.cluster import KMeans
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(X)
        self.msg = ""
        for i in range(len(y_kmeans)):
            self.msg += "Cluster No :" + str(y_kmeans[i]) + "     Value  :" + str(y[i]) + "\n"

    def disp_result(self):
        return self.msg

class HC():
    def __init__(self,dataset):

        X = dataset.iloc[:, [3, 4]].values
        y = dataset.iloc[:, 3].values
        import scipy.cluster.hierarchy as sch
        dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
        from sklearn.cluster import AgglomerativeClustering
        hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
        y_hc = hc.fit_predict(X)
        self.msg = ""
        for i in range(len(y_hc)):
            self.msg += "Cluster No :" + str(y_hc[i]) + "     Value  :" + str(y[i]) + "\n"

    def disp_result(self):
        return self.msg