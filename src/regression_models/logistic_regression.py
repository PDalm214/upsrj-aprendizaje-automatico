# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Aprendizaje automático
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC02
# Archivo: logistic_regression.py
# Descripción: Definición de clase LogisticRegression
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

from regression_models.data_source import DataSource as ds

class LogisticRegressionCompare:
    def __init__(self, url: str, base: str, out: str):
        self.source = ds(url, churn=True)

        posibles_columnas = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
        cols_validas = [c for c in posibles_columnas if c in self.source.data.columns]

        if len(cols_validas) == 0:
            cols_validas = [c for c in self.source.data.columns if c != base]

        self.x = np.asarray(self.source.data[cols_validas])
        self.y = np.asarray(self.source.data[base])

        base_col = base
        if base_col not in self.source.data.columns:
            posibles = [c for c in self.source.data.columns if c.lower() == base.lower()]
            if len(posibles) > 0:
                base_col = posibles[0]
            else:
                raise KeyError(f"No se encontró la columna base '{base}' en el DataFrame: {self.source.data.columns.tolist()}")

        self.y = np.asarray(self.source.data[base_col])

        self.std_scaler, self.x_std = self.standarize(x=self.x)
        self.d = self.prepare_data(x=self.x_std, y=self.y, prc=0.2, random_state=4)
        self.m = self.create_model()
        self.train_model(self.m, self.d)
        self.plot_model_and_predict(self.m, index=cols_validas, x=self.d[1], 
                                    y=self.d[3], out=os.path.join(out, "logistic_regression_churn_coefficients.png"))
        
    def standarize(self, x: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
        std_scaler = StandardScaler()
        x_std = std_scaler.fit_transform(x)
        return std_scaler, x_std
    
    def prepare_data(self, x:np.ndarray, y:np.ndarray, prc: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=prc, random_state=random_state)
        return (x_train, x_test, y_train, y_test)

    def create_model(self) -> LogisticRegression:
        return LogisticRegression(max_iter=1000)
    
    def train_model(self, model: LogisticRegression, data: np.ndarray) -> None:
        x_train, x_test, y_train, y_test = data
        y_train = y_train.ravel() if y_train.ndim > 1 else y_train
        model.fit(x_train, y_train)
    
    def plot_model_and_predict(self, model: LogisticRegression, index, x: np.ndarray, y:np.ndarray, out: str) -> None:
        try:
            yhat_prob = model.predict_proba(x)
            coefficients = pd.Series(model.coef_[0], index=index)
            coefficients.sort_values().plot(kind='barh')
            plt.title("Feature Coefficients in Logistic Regression Churn Model")
            plt.xlabel("Coefficient Value")
            plt.savefig(out)
            plt.close()
            log_loss(y, yhat_prob)
            print(f"Se creó gráfico de coeficientes en {out}")
        except Exception as e:
            print(f"Error: no se pudo crear gráfico de coeficientes en {out}: {e}")