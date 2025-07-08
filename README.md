# Technosp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

class MentalHealthData:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.df = self.generate_data()

    def generate_data(self):
        """Generate synthetic mental health dataset"""
        np.random.seed(42)
        data = {
            'sleep_hours': np.clip(np.random.normal(6.8, 1.2, self.n_samples), 3, 10),
            'work_stress': np.random.randint(1, 11, self.n_samples),
            'social_connections': np.random.poisson(3, self.n_samples),
            'physical_activity': np.random.binomial(7, 0.4, self.n_samples),
            'screen_time': np.clip(np.random.normal(5.5, 2.1, self.n_samples), 1, 16),
            'healthy_meals': np.random.randint(0, 7, self.n_samples),
            'stress_level': np.random.randint(0, 2, self.n_samples)  # 0=Low, 1=High
        }
        df = pd.DataFrame(data)

        # Add derived features
        df['sleep_quality'] = df['sleep_hours'] / 10
        df['productivity_score'] = (df['physical_activity'] + df['social_connections']) / 14
        df['digital_detox'] = (16 - df['screen_time']) / 15
        return df

    def get_correlation_matrix(self):
        """Generate correlation matrix visualization"""
        plt.figure(figsize=(10, 8))
        corr = self.df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Mental Health Factors Correlation')
        # return plt.gcf() # Removed as GUI canvas is not used
        plt.show() # Use plt.show() to display the plot in Colab

class StressPredictor:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.accuracy = None
        self.train_model()

    def train_model(self):
        """Train Random Forest classifier"""
        X = self.data.df.drop('stress_level', axis=1)
        y = self.data.df['stress_level']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test) # Corrected from y_test to X_test
        self.accuracy = accuracy_score(y_test, y_pred)

    def predict_stress(self, input_data):
        """Predict stress level from input features"""
        try:
            input_df = pd.DataFrame([input_data])
            proba = self.model.predict_proba(input_df)[0][1]  # Probability of high stress
            return proba
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
if __name__ == "__main__":
    # Initialize data and predictor
    data = MentalHealthData()
    predictor = StressPredictor(data)

    # Display model accuracy
    print(f"Model Accuracy: {predictor.accuracy:.2%}")

    # Display correlation matrix
    data.get_correlation_matrix()

    # Display feature importance
    feature_imp = pd.Series(
        predictor.model.feature_importances_,
        index=data.df.drop('stress_level', axis=1).columns
    ).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    feature_imp.plot(kind='barh', ax=ax)
    ax.set_title('Feature Importance for Stress Prediction')
    plt.tight_layout()
    plt.show()

    # Example prediction
    sample_input = {
        'sleep_hours': 7.5,
        'work_stress': 8,
        'social_connections': 2,
        'physical_activity': 3,
        'screen_time': 7.0,
        'healthy_meals': 5,
        'sleep_quality': 0.75,
        'productivity_score': (3 + 2) / 14,
        'digital_detox': (16 - 7.0) / 15
    }
    stress_probability = predictor.predict_stress(sample_input)
    print(f"\nSample input stress probability: {stress_probability:.2%}")