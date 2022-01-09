import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def show_image(bits):
    bits = np.reshape(bits, (28, 28))
    plt.imshow(bits, cmap='binary')
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', version=1, as_frame=True)
    print(mnist.keys())

    X, y = pd.DataFrame(mnist['data']), pd.Series(mnist['target']).astype(dtype=np.uint8)
    X.info()
    print(X.shape, y.shape)

    show_image(X.iloc[0].values)
    train_size = 60000
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = KNeighborsClassifier(n_jobs=-1)
    parameters = {
        'n_neighbors': (1, 3, 5, 10, 20),
        'weights': ('uniform', 'distance')
    }
    search = GridSearchCV(model, parameters, cv=3,
                          scoring='accuracy',
                          verbose=4, return_train_score=True)
    search.fit(X_train.values, y_train)
    print(search.best_params_)
    cv_res = search.cv_results_
    for mean_score, params in zip(cv_res['mean_test_score'], cv_res['params']):
        print(mean_score, params)
    model = search.best_estimator_

    # train_res = cross_val_predict(model, X_train, y_train, cv=3)
    # print(list(y_train)[:10])
    # print(list(train_res)[:10])
    # print(classification_report(y_train, train_res, digits=3))
    # model.fit(X_train, y_train)

    test_res = model.predict(X_test.values)
    print(list(y_test)[:10])
    print(list(test_res)[:10])
    print(classification_report(y_test, test_res, digits=3))
