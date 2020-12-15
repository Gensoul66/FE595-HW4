import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston,load_wine,load_iris
import matplotlib.pyplot as plt

def main():

    # Question1

    # Load the data boston
    boston = load_boston()
    boston_x, boston_y = load_boston(return_X_y=True)

    # Regression on the value of the house
    lm = LinearRegression()
    linear_reg = lm.fit(X=boston_x, y=boston_y)
    linear_cof = linear_reg.coef_
    feature_names = boston.feature_names

    # abs cof
    linear_abs_cof = np.abs(linear_cof)
    print("Question1:" + "\n")
    print("Predictor" + "\t" + "Slope" + "\n")

    # out put the result
    for i in range(len(linear_abs_cof)):
        index = np.argmax(linear_abs_cof)
        linear_abs_cof[index] = 0
        str1 = str(feature_names[index]) + "\t\t"
        str2 = str(linear_cof[index]) + "\n"
        print(str1 + str2)
    print("===========================================" + "\n")

    # Question2

    # load data wine&iris
    wine = load_wine()
    iris = load_iris()

    wine_x, wine_y = load_wine(return_X_y=True)
    iris_x, iris_y = load_iris(return_X_y=True)

    # seeking the best k
    distortion_wine = []
    distortion_iris = []
    k = [1, 2, 3, 4, 5, 6, 7]

    for i in k:
        kmeans_wine = KMeans(n_clusters=i)
        kmeans_iris = KMeans(n_clusters=i)
        kmeans_wine.fit(wine_x)
        kmeans_iris.fit(iris_x)
        distortion_wine.append(kmeans_wine.inertia_)
        distortion_iris.append(kmeans_iris.inertia_)

    # WineEH plot
    plt.plot(k, distortion_wine, marker='o')
    plt.xlabel('Number Of Clusters')
    plt.ylabel('Distortion')
    plt.title("DataSet: Wine")
    plt.show()
    plt.savefig('Wine_EH.png')

    # irisEH plot
    plt.plot(k, distortion_iris, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title("DataSet: Iris")
    plt.show()
    plt.savefig('Iris_EH.png')


    # Verify the assumption

    print("Question2: " + "\n")
    print("There are three classes in the wine data set and iris data set"+"\n")
    print(wine.target_names)
    print(iris.target_names)


if __name__ == "__main__" :
    main()