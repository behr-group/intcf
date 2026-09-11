import numpy as np
from intcf import CausalForest
from sklearn.preprocessing import StandardScaler

from .base import BaselineModel


class IntRFBaseline(BaselineModel):
    def __init__(self, honest: bool = True):
        super().__init__(hpo=False)
        self.honest = honest

    def _scale_x(self, X_tr, X_te):
        scaler_x = StandardScaler().fit(X_tr)
        return (scaler_x.transform(X_tr).astype("float32"), scaler_x.transform(X_te).astype("float32"), scaler_x)

    def _scale_y(self, y_tr):
        scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
        y_tr_s = scaler_y.transform(y_tr.reshape(-1, 1)).ravel().astype("float32")
        return y_tr_s, scaler_y

    def _unscale_res(self, res, scaler_y):
        return res * scaler_y.scale_[0] if scaler_y is not None else res

    def _get_model(self, X, t, y):
        model = CausalForest(n_estimators=100)
        return model

    def estimate_ate(self, X, t, y):
        X, _, _ = self._scale_x(X, X)
        y, scaler_y = self._scale_y(y)

        model = self._get_model(X, t, y)
        model.fit(X=X, T=t, y=y, oob_reeval=self.honest)
        if self.honest:
            ate_pred = np.mean([tree.root.mean_predict for tree in model.estimators])
        else:
            ate_pred = np.mean(model.predict(covariates=X))
        return self._unscale_res(ate_pred, scaler_y)

    def estimate_cate(self, X_train, t_train, y_train, X_test):
        X_train, X_test, _ = self._scale_x(X_train, X_test)
        y_train, scaler_y = self._scale_y(y_train)

        model = self._get_model(X_train, t_train, y_train)
        model.fit(X=X_train, T=t_train, y=y_train, oob_reeval=self.honest)
        cate_pred = model.predict(covariates=X_test)
        return self._unscale_res(cate_pred, scaler_y)

    def estimate_ci(self, X_train, t_train, y_train, X_test, alpha=0.05):
        raise ValueError("The model does not support confidence intervals.")
