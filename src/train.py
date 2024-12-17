import lightgbm as lgb
from sklearn.metrics import classification_report
from joblib import dump


def train_model( X_train, X_test, y_train, y_test):        
        model = lgb.LGBMClassifier(n_estimators=10,max_depth=3)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)
        # predict the classes
        y_pred = model.predict(X_test)
        # print the classification report
        print(classification_report(y_test, y_pred))
        dump(model, 'trained_model.joblib')
        booster = model.booster_
        trees=booster.dump_model()['tree_info']
        nofcl=model.n_classes_
        return trees,nofcl