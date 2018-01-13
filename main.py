import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm

def get_data():
    data_dir = "./datasets/Skin_NonSkin.csv"
    data = pd.read_csv(data_dir)
    X = np.array(data[["B", "G", "R"]])
    y = np.array(data["y"])
    # print(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape,y_train.shape)
    print(X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test

def draw_points(X_train, y_train, X_test, y_test):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot_points(X,y,c1,c2,title):
        one = X[np.argwhere(y==1)]
        zero = X[np.argwhere(y==2)]
        ax.scatter([s[0][0] for s in one], [s[0][1] for s in one], [s[0][2] for s in one], c=c1, edgecolors='k', alpha=0.3, label="y=1")
        ax.scatter([s[0][0] for s in zero], [s[0][1] for s in zero], [s[0][2] for s in zero], c=c2, edgecolors='k', alpha=0.3, label="y=0")
        ax.set_xlabel('Feature a')
        ax.set_ylabel('Feature b')
        ax.set_zlabel('Feature c')
        ax.legend(loc=2)
        ax.set_title(title)

    # plot_points(X_train, y_train, c1='red', c2='blue', title='')
    plot_points(X_test, y_test, c1='yellow', c2='green', title='')
    plt.show()

def draw_plot_2D(X_train, y_train, X_test, y_test):
    fig, axes = plt.subplots(2, 2)

    def plot_points(X, y, c1, c2):
        cs = [[125, 127], [130, 132], [168, 171], [223, 225]]
        for ax, c in zip(axes.flat, cs):
            one = X[np.where((X[:, 2] > c[0]) & (X[:, 2] < c[1]) & (y == 1))]
            zero = X[np.where((X[:, 2] > c[0]) & (X[:, 2] < c[1]) & (y == 2))]
            ax.scatter([s[0] for s in zero], [s[1] for s in zero], c=c2, edgecolors='k', alpha=0.5)
            ax.scatter([s[0] for s in one], [s[1] for s in one], c=c1, edgecolors='k', alpha=0.5)
            ax.set_xlabel('Feature a')
            ax.set_ylabel('Feature b')
            ax.set_title('c(%d, %d)'%(c[0], c[1]))

    plot_points(X_train, y_train, c1='red', c2='blue')
    # plot_points(X_test, y_test, c1='yellow', c2='green')
    fig.tight_layout()
    plt.show()

def write_csv(results, csv_name):
    save = pd.DataFrame(data=results)
    save.to_csv(csv_name, index=False)

def print_predict(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    v = []
    if(hasattr(clf, 'support_vectors_')):
        v = clf.support_vectors_
        # print(v)
    y_test_pred = clf.predict(X_test)
    num_correct = np.sum(y_test_pred == y_test)
    acc = float(num_correct) / len(y_test)
    print('Test data got %d / %d = %f accuracy' % (num_correct, len(y_test), acc))
    return v

def model_svm(X_train, y_train, X_test, y_test):
    def get_parameters():
        parameters = {'C':[], 'gamma': []}
        parameters['C'] = [float(i/10) for i in range(1, 10)]
        for i in range(1, 11):
            parameters['C'].append(i)
        for j in range(1, 101):
            a = float((j-1)/100)
            b = float(j/100)
            rand = (b - a)*np.random.sample() + a # 取[0.1, 0.2)之间的值, 以此类推
            parameters['gamma'].append(rand)
        return parameters

    # 交叉验证5次得出训练模型，选出最好的超参数对test进行训练
    best_score = 0.0
    best_parameters = {}
    data = {'C': [], 'gamma': [], 'Accuracy mean': []}
    parameters = get_parameters()
    parameters = {'C':[0.1, 0.5, 1, 10], 'gamma': [0.017928, 0.029023, 0.322087, 0.335604]} # 测试用
    print(parameters)
    for C in parameters['C']:
        for gamma in parameters['gamma']:
            clf = svm.SVC(C=C, gamma=gamma)
            score = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=2)
            this_score = np.mean(score)
            data['C'].append(C)
            data['gamma'].append(gamma)
            data['Accuracy mean'].append(this_score)
            print('gamma: %f, C: %f, best score: %f, Accuracy mean: %f' % (gamma, C, best_score, this_score,))
            if(this_score > best_score):
                best_parameters['gamma'] = gamma
                best_parameters['C'] = C
                best_score = this_score
    print(best_parameters)
    write_csv(data, 'csv_result_svm.csv')
    clf = svm.SVC(C=best_parameters['C'], gamma=best_parameters['gamma'])
    v = print_predict(clf, X_train, y_train, X_test, y_test)
    return v

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data()
    v = []
    # model_knn(X_train, y_train, X_test, y_test)
    v = model_svm(X_train, y_train, X_test, y_test)
    print(v.shape)
    # draw_points(X_train, y_train, X_test, y_test)
    # draw_plot_result(v)
    # draw_coutour()
    draw_plot_2D(X_train, y_train, X_test, y_test)