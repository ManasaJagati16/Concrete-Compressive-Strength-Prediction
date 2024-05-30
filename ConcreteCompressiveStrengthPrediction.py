
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


class ConcreteStrengthPredictor:
    def __init__(self, data_file, eda_folder, results_folder):
        self.data = pd.read_excel(data_file)
        self.clean_data()
        self.eda_folder = eda_folder
        self.results_folder = results_folder
        self.visualize_data()


    def clean_data(self):
        # Add 1 to each value to avoid log transformation issues
        self.data += 1
        self.data = np.log(self.data)

    def save_visualization(self, fig, filename):
        # Save visualizations to the specified 'EDA' folder
        fig.savefig(os.path.join(self.eda_folder, filename))

    def visualize_data(self):
        # Visualize data using Matplotlib
        fig = plt.figure(figsize=(15, 15), facecolor='white')
        plotnumber = 1

        for column in self.data.columns:
            ax = fig.add_subplot(4, 3, plotnumber)
            sns.distplot(self.data[column])
            ax.set_xlabel(column, fontsize=10)
            plotnumber += 1

        # Save the data distribution visualization
        self.save_visualization(fig, 'data_distribution.png')
        

        # Data Transformation Visualization
        fig = plt.figure(figsize=(15, 15), facecolor='white')
        plotnumber = 1

        X = self.data.drop(['Concrete compressive strength(MPa, megapascals) '], axis=1)

        for column in X.columns:
            ax = fig.add_subplot(4, 3, plotnumber)
            sns.distplot(X[column])
            ax.set_xlabel(column, fontsize=10)
            plotnumber += 1

        # Save the data transformation visualization
        self.save_visualization(fig, 'data_transformation.png')
        

        # Outliers Visualization
        fig = plt.figure(figsize=(15, 15), facecolor='white')
        plotnumber = 1

        for column in X.columns:
            ax = fig.add_subplot(4, 3, plotnumber)
            sns.boxplot(X[column])
            ax.set_xlabel(column, fontsize=10)
            plotnumber += 1

        # Save the outliers visualization
        self.save_visualization(fig, 'outliers.png')
        

        # Relationship with Target Visualization
        fig = plt.figure(figsize=(20, 20), facecolor='white')
        plotnumber = 1

        for column in X.columns:
            ax = fig.add_subplot(4, 3, plotnumber)
            sns.scatterplot(x=X[column], y=self.data['Concrete compressive strength(MPa, megapascals) '], ax=ax)
            ax.set_xlabel(column, fontsize=10)
            plotnumber += 1

        # Save the relationship visualization
        self.save_visualization(fig, 'relationship_with_target.png')
        

        # Correlation Heatmap Visualization
        fig = plt.figure(figsize=(8, 8))
        sns.heatmap(X.corr().abs(), vmin=-0.5, vmax=1, annot=True)

        self.save_visualization(fig, 'correlation.png')
        

    def split_and_scale_data(self):
        X = self.data.drop(['Concrete compressive strength(MPa, megapascals) '], axis=1)
        y = self.data['Concrete compressive strength(MPa, megapascals) ']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        return X_train, X_test, y_train, y_test
    
    
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        print("Linear Regression Train Score:", lr.score(X_train, y_train))
        print("Linear Regression Test Score:", lr.score(X_test, y_test))
        return lr

    
    def train_lasso_regression(self, X_train, y_train, X_test, y_test):
        lassocv = LassoCV(alphas=None, cv=10, max_iter=10000)
        lassocv.fit(X_train, y_train)
        lasso = Lasso(alpha=lassocv.alpha_)
        lasso.fit(X_train, y_train)

        print("Lasso Regression Train Score:", lasso.score(X_train, y_train))
        print("Lasso Regression Test Score:", lasso.score(X_test, y_test))
        return lasso

    

    def train_random_forest_regression(self, X_train, y_train, X_test, y_test):
        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)
        print("Random Forest Regression Train Score:", rfr.score(X_train, y_train))
        print("Random Forest Regression Test Score:", rfr.score(X_test, y_test))
        return rfr


    def train_decision_tree_regressor(self, X_train, y_train, X_test, y_test):
        dtr = DecisionTreeRegressor()
        dtr.fit(X_train, y_train)
        print("Decision Tree Regressor Train Score:", dtr.score(X_train, y_train))
        print("Decision Tree Regressor Test Score:", dtr.score(X_test, y_test))
        return dtr


    def hyperparameter_tuning_decision_tree(self, X_train, y_train):
        grid_params = {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'splitter': ['best', 'random'],
            'max_depth': [3, 5, 7, 9, 10],
            'min_samples_split': [1, 2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3, 4, 5]
        }

        grid_search = GridSearchCV(DecisionTreeRegressor(), grid_params, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        print("Best parameters for Decision Tree Regressor:", grid_search.best_params_)
        print("Best score for Decision Tree Regressor:", grid_search.best_score_)

        best_dtr = DecisionTreeRegressor(**grid_search.best_params_)
        best_dtr.fit(X_train, y_train)

        return best_dtr

    def train_random_forest_regression_tuned(self, X_train, y_train, X_test, y_test):
        best_dtr = self.hyperparameter_tuning_decision_tree(X_train, y_train)

        rfr_tuned = RandomForestRegressor(**best_dtr.get_params())
        rfr_tuned.fit(X_train, y_train)
        print("Random Forest Regression (Tuned) Train Score:", rfr_tuned.score(X_train, y_train))
        print("Random Forest Regression (Tuned) Test Score:", rfr_tuned.score(X_test, y_test))
        return rfr_tuned

    
    def hyperparameter_tuning_ada_boost(self, X_train, y_train):
        base_estimator = DecisionTreeRegressor()

        grid_params = {
            'n_estimators': [40, 50, 80, 100],
            'learning_rate': [0.01, 0.1, 0.05, 0.5, 1, 10],
            'loss': ['linear', 'square', 'exponential']
        }

        grid_search = GridSearchCV(AdaBoostRegressor(base_estimator=base_estimator), grid_params, cv=5, n_jobs=-1,
                                   verbose=1)
        grid_search.fit(X_train, y_train)

        print("Best parameters for Ada Boost Regressor:", grid_search.best_params_)
        print("Best score for Ada Boost Regressor:", grid_search.best_score_)

        best_ada = AdaBoostRegressor(base_estimator=base_estimator, **grid_search.best_params_)
        best_ada.fit(X_train, y_train)

        return best_ada

    def train_ada_boost_regressor(self, X_train, y_train, X_test, y_test):
        # Base AdaBoost model
        ada_base = AdaBoostRegressor(base_estimator=DecisionTreeRegressor())
        ada_base.fit(X_train, y_train)
        print("Ada Boost Regressor Train Score (Base):", ada_base.score(X_train, y_train))
        print("Ada Boost Regressor Test Score (Base):", ada_base.score(X_test, y_test))


        # Tuned AdaBoost model
        best_ada = self.hyperparameter_tuning_ada_boost(X_train, y_train)
        ada_tuned = AdaBoostRegressor(base_estimator=best_ada)
        ada_tuned.set_params(**best_ada.get_params())
        ada_tuned.fit(X_train, y_train)


        ada_tuned.fit(X_train, y_train)
        print("Ada Boost Regressor Train Score (Tuned):", ada_tuned.score(X_train, y_train))
        print("Ada Boost Regressor Test Score (Tuned):", ada_tuned.score(X_test, y_test))

        return ada_base, ada_tuned

    def train_xgboost_regressor(self, X_train, y_train, X_test, y_test):
        xgb = XGBRegressor(booster='gbtree', learning_rate=0.1, max_depth=7, n_estimators=200)
        xgb.fit(X_train, y_train)
        print("XgBoost Regressor Train Score:", xgb.score(X_train, y_train))
        print("XgBoost Regressor Test Score:", xgb.score(X_test, y_test))
        return xgb


    def calculate_model_score(self, model, X_test, y_test):
        return model.score(X_test, y_test)


    def run_models(self):
        X_train, X_test, y_train, y_test = self.split_and_scale_data()

        # Training models
        lr = self.train_linear_regression(X_train, y_train, X_test, y_test)
        lasso = self.train_lasso_regression(X_train, y_train, X_test, y_test)
        rfr = self.train_random_forest_regression(X_train, y_train, X_test, y_test)
        dtr = self.train_decision_tree_regressor(X_train, y_train, X_test, y_test)
        ada_base, ada_tuned = self.train_ada_boost_regressor(X_train, y_train, X_test, y_test)
        xgb = self.train_xgboost_regressor(X_train, y_train, X_test, y_test)

        # Visualize and save models accuracy
        models_df = pd.DataFrame({
            'Model': ['Linear Regression', 'Lasso Regression', 'Random Forest', 'Decision Tree', 'Ada Boost', 'XgBoost'],
            'Score': [
                self.calculate_model_score(lr, X_test, y_test),
                self.calculate_model_score(lasso, X_test, y_test),
                self.calculate_model_score(rfr, X_test, y_test),
                self.calculate_model_score(dtr, X_test, y_test),
                self.calculate_model_score(ada_tuned, X_test, y_test),  # Use the tuned AdaBoost model
                self.calculate_model_score(xgb, X_test, y_test)
            ]
        })

        
        self.visualize_and_save_models_accuracy(models_df)

        # Save the machine learning results
        self.save_ml_results(X_train, X_test, y_train, y_test, lr, lasso, rfr, dtr, ada_base, ada_tuned, xgb)


        
    def save_ml_results(self, X_train, X_test, y_train, y_test, lr, lasso, rfr, dtr, ada_base, ada_tuned, xgb):
        # Save the results to the specified 'Results' folder
        with open(os.path.join(self.results_folder, 'linear_regression_results.txt'), 'w') as f:
            f.write("Linear Regression Train Score: {}\n".format(lr.score(X_train, y_train)))
            f.write("Linear Regression Test Score: {}\n".format(lr.score(X_test, y_test)))

        with open(os.path.join(self.results_folder, 'lasso_regression_results.txt'), 'w') as f:
            f.write("Lasso Regression Train Score: {}\n".format(lasso.score(X_train, y_train)))
            f.write("Lasso Regression Test Score: {}\n".format(lasso.score(X_test, y_test)))

        with open(os.path.join(self.results_folder, 'random_forest_results.txt'), 'w') as f:
            f.write("Random Forest Regression Train Score: {}\n".format(rfr.score(X_train, y_train)))
            f.write("Random Forest Regression Test Score: {}\n".format(rfr.score(X_test, y_test)))

        with open(os.path.join(self.results_folder, 'decision_tree_results.txt'), 'w') as f:
            f.write("Decision Tree Regression Train Score: {}\n".format(dtr.score(X_train, y_train)))
            f.write("Decision Tree Regression Test Score: {}\n".format(dtr.score(X_test, y_test)))

        with open(os.path.join(self.results_folder, 'ada_boost_results.txt'), 'w') as f:
            f.write("AdaBoost Regression Train Score (Base): {}\n".format(ada_base.score(X_train, y_train)))
            f.write("AdaBoost Regression Test Score (Base): {}\n".format(ada_base.score(X_test, y_test)))
            f.write("AdaBoost Regression Train Score (Tuned): {}\n".format(ada_tuned.score(X_train, y_train)))
            f.write("AdaBoost Regression Test Score (Tuned): {}\n".format(ada_tuned.score(X_test, y_test)))

        with open(os.path.join(self.results_folder, 'xgboost_results.txt'), 'w') as f:
            f.write("XGBoost Regression Train Score: {}\n".format(xgb.score(X_train, y_train)))
            f.write("XGBoost Regression Test Score: {}\n".format(xgb.score(X_test, y_test)))

   
    def visualize_and_save_models_accuracy(self, models_df):
        # Convert 'Score' column to numeric
        models_df['Score'] = pd.to_numeric(models_df['Score'])

        # Sort DataFrame by 'Score' column
        models_df = models_df.sort_values(by='Score', ascending=False)

        # Create the bar plot using matplotlib directly
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.bar(models_df['Model'], models_df['Score'])
        
        # Add value annotations on top of each bar
        for i, v in enumerate(models_df['Score']):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

        # Set plot labels and display
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Scores')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility

        # Save the ML models accuracy
        self.save_visualization(fig, 'Models_Accuracy.png')


if __name__ == "__main__":
    # Manually provide the paths to 'EDA' and 'Results' folders
    eda_folder_path = "Data_601_Group_6_Project/EDA"
    results_folder_path = "Data_601_Group_6_Project/Results"

    training_file = "Data_601_Group_6_Project/Input_File/Concrete_Data.xls"

    # Use the correct file path directly
    predictor = ConcreteStrengthPredictor(training_file, eda_folder_path, results_folder_path)
    predictor.run_models()









