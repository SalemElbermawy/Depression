import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error


import plotly.express as px
import plotly.graph_objects as go

class Data():
    def __init__(self, q1, q2, q3, q4, q5, q6, q7, q8, q9, gen, age):
        self.gen = gen
        self.age = age
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.q5 = q5
        self.q6 = q6
        self.q7 = q7
        self.q8 = q8
        self.q9 = q9

        # ================== Load Data ==================
        data = pd.read_csv("data.csv")
        target = data["PHQ9 score"]

        data.drop(["Institute", "PHQ9 score"], inplace=True, axis="columns")

        # ================== Encode Gender ==================
        model_one = LabelEncoder()
        gender = model_one.fit_transform(data["Gender"])
        data_gender = pd.DataFrame(gender, columns=["gender"])

        data = pd.concat([data_gender, data], axis=1)
        data.drop("Gender", axis="columns", inplace=True)

        # ================== Train/Test Split ==================
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, target, test_size=0.2, random_state=42
        )

        # ================== Model ==================
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42
        )

        # ================== Hyperparameter Search ==================
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.1, 0.5, 1],
            "reg_lambda": [0.5, 1, 2]
        }

        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_dist,
            n_iter=30,
            cv=3,
            scoring="neg_mean_squared_error",
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(self.X_train, self.y_train)

        # ================== Best Model ==================
        self.best_model = random_search.best_estimator_

    def local_evaluate(self):
        y_pred = self.best_model.predict(self.X_test)

        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        sample = self.X_test.iloc[[0]]
        pred_score = self.best_model.predict(sample)[0]
        
        self.y_pred = y_pred
        self.mse = mse
        self.r2 = r2 

        return(f"📊 MSE:  {round(mse,3)} \n\n📊 RMSE :  {round(np.sqrt(mse),3)} \n\n📊 R² Score: {round(r2,3)} \n\n📊 Predicted PHQ9 score for sample   :  {pred_score:.2f}")

       
        

    def special_evaluate(self):
        if self.gen == "Male":
            self.gen = 1
        else:
            self.gen = 0

        test_data = pd.DataFrame({
            "gender": [self.gen],
            "Age": [self.age],
            "q1": [self.q1],
            "q2": [self.q2],
            "q3": [self.q3],
            "q4": [self.q4],
            "q5": [self.q5],
            "q6": [self.q6],
            "q7": [self.q7],
            "q8": [self.q8],
            "q9": [self.q9],
        })

        self.test_predict = self.best_model.predict(test_data)

    def present_test(self):
        pred = self.test_predict[0]
        if pred > 27:
            return 27
        elif pred < 0:
            return 0
        else:
            return pred
    def plot_accuracy(self):
        fig = go.Figure(data=[
            go.Bar(name="R² Score", x=["R²"], y=[self.r2]),
            go.Bar(name="RMSE", x=["RMSE"], y=[np.sqrt(self.mse)])
        ])
        fig.update_layout(
            title="Model Performance Metrics",
            barmode="group"
        )
        return fig
    def plot_predictions(self):
        fig = px.scatter(
            x=self.y_test,
            y=self.y_pred,
            labels={"x": "True PHQ9 Score", "y": "Predicted PHQ9 Score"},
            title="True vs Predicted PHQ9 Scores"
        )
        fig.add_shape(
            type="line", x0=self.y_test.min(), y0=self.y_test.min(),
            x1=self.y_test.max(), y1=self.y_test.max(),
            line=dict(color="red", dash="dash")
        )
        return fig


