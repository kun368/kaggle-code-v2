from sklearn.datasets import make_moons
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    df = make_moons(n_samples=10000, noise=0.4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(df[0], df[1], test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape)

    model = DecisionTreeClassifier()
    parameters = {
        'max_leaf_nodes': (4, 8, 16, 32, 64),
        'max_depth': (2, 4, 6, 8, 10),
    }
    search = GridSearchCV(model, parameters, cv=3,
                          scoring='accuracy',
                          verbose=3, return_train_score=True)
    search.fit(X_train, y_train)
    print('best_params', search.best_params_)

    model = search.best_estimator_
    print(classification_report(y_test, model.predict(X_test), digits=4))
