import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sqlite3
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinancialMLModels:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_names = []
        
    def load_data(self, source='database'):
        """Load processed data"""
        if source == 'database':
            conn = sqlite3.connect('data/processed/transactions.db')
            df = pd.read_sql_query("SELECT * FROM transactions", conn)
            conn.close()
        else:
            df = pd.read_csv('data/processed/processed_transactions.csv')
        
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        features_df = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['Category', 'Account', 'DayName', 'MonthName']
        
        for col in categorical_columns:
            if col in features_df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    features_df[f'{col}_encoded'] = self.encoders[col].fit_transform(features_df[col].fillna('Unknown'))
                else:
                    # Handle unseen categories
                    unique_vals = set(features_df[col].fillna('Unknown').unique())
                    known_vals = set(self.encoders[col].classes_)
                    new_vals = unique_vals - known_vals
                    
                    if new_vals:
                        # Add new categories to encoder
                        self.encoders[col].classes_ = np.append(self.encoders[col].classes_, list(new_vals))
                    
                    features_df[f'{col}_encoded'] = features_df[col].fillna('Unknown').apply(
                        lambda x: self.encoders[col].transform([x])[0] if x in self.encoders[col].classes_ else -1
                    )
        
        # Select numerical features
        numerical_features = [
            'AbsAmount', 'Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 
            'IsWeekend', 'IsExpense', 'MerchantFrequency', 'CategoryFrequency',
            'DaysSinceLastCategoryTransaction'
        ]
        
        # Add encoded categorical features
        encoded_features = [col for col in features_df.columns if col.endswith('_encoded')]
        
        # Combine all features
        self.feature_names = numerical_features + encoded_features
        
        # Select only available features
        available_features = [col for col in self.feature_names if col in features_df.columns]
        X = features_df[available_features].fillna(0)
        
        return X, features_df
    
    def train_expense_predictor(self, df):
        """Train model to predict monthly expenses"""
        print("🤖 Training expense prediction model...")
        
        # Aggregate monthly expenses by category
        df['YearMonth'] = df['DateTime'].dt.to_period('M')
        monthly_data = []
        
        for period in df['YearMonth'].unique():
            period_data = df[df['YearMonth'] == period]
            
            # Basic features
            features = {
                'Month': period.month,
                'Year': period.year,
                'TotalTransactions': len(period_data),
                'TotalExpense': abs(period_data[period_data['Amount'] < 0]['Amount'].sum()),
                'AvgTransactionSize': period_data['AbsAmount'].mean(),
                'WeekendTransactions': len(period_data[period_data['IsWeekend'] == 1]),
            }
            
            # Category-wise expenses
            for category in df['Category'].unique():
                cat_expense = abs(period_data[
                    (period_data['Category'] == category) & (period_data['Amount'] < 0)
                ]['Amount'].sum())
                features[f'Expense_{category.replace(" ", "_").replace("&", "and")}'] = cat_expense
            
            monthly_data.append(features)
        
        monthly_df = pd.DataFrame(monthly_data)
        
        # Prepare features for prediction (predict next month's total expense)
        X = monthly_df.drop(['TotalExpense'], axis=1)
        y = monthly_df['TotalExpense']
        
        if len(X) < 3:  # Need at least 3 months of data
            print("⚠️ Not enough data for expense prediction model")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scalers['expense_predictor'] = StandardScaler()
        X_train_scaled = self.scalers['expense_predictor'].fit_transform(X_train)
        X_test_scaled = self.scalers['expense_predictor'].transform(X_test)
        
        # Train model
        self.models['expense_predictor'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['expense_predictor'].fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.models['expense_predictor'].predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Expense predictor trained - RMSE: ${rmse:.2f}, R²: {r2:.3f}")
        
        return {
            'model': self.models['expense_predictor'],
            'scaler': self.scalers['expense_predictor'],
            'features': X.columns.tolist(),
            'rmse': rmse,
            'r2': r2
        }
    
    def train_anomaly_detector(self, df):
        """Train anomaly detection model"""
        print("🔍 Training anomaly detection model...")
        
        # Prepare features
        X, _ = self.prepare_features(df)
        
        # Add amount-based features for anomaly detection
        amount_features = pd.DataFrame({
            'Amount': df['Amount'],
            'AbsAmount': df['AbsAmount'],
            'AmountZScore': (df['AbsAmount'] - df['AbsAmount'].mean()) / df['AbsAmount'].std()
        })
        
        X_anomaly = pd.concat([X, amount_features], axis=1).fillna(0)
        
        # Scale features
        self.scalers['anomaly_detector'] = StandardScaler()
        X_scaled = self.scalers['anomaly_detector'].fit_transform(X_anomaly)
        
        # Train Isolation Forest
        self.models['anomaly_detector'] = IsolationForest(
            contamination=0.05,  # Expect 5% anomalies
            random_state=42
        )
        self.models['anomaly_detector'].fit(X_scaled)
        
        # Get anomaly scores
        anomaly_scores = self.models['anomaly_detector'].decision_function(X_scaled)
        anomaly_labels = self.models['anomaly_detector'].predict(X_scaled)
        
        print(f"✅ Anomaly detector trained - Found {(anomaly_labels == -1).sum()} anomalies")
        
        return {
            'model': self.models['anomaly_detector'],
            'scaler': self.scalers['anomaly_detector'],
            'features': X_anomaly.columns.tolist()
        }
    
    def train_category_predictor(self, df):
        """Train model to predict transaction category"""
        print("🏷️ Training category prediction model...")
        
        # Prepare features (excluding category-related features)
        feature_cols = [
            'AbsAmount', 'Hour', 'DayOfWeek', 'IsWeekend',
            'MerchantFrequency', 'DaysSinceLastCategoryTransaction'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].fillna(0)
        y = df['Category']
        
        # Encode target
        self.encoders['category_target'] = LabelEncoder()
        y_encoded = self.encoders['category_target'].fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        self.scalers['category_predictor'] = StandardScaler()
        X_train_scaled = self.scalers['category_predictor'].fit_transform(X_train)
        X_test_scaled = self.scalers['category_predictor'].transform(X_test)
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        self.models['category_predictor'] = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.models['category_predictor'].fit(X_train_scaled, y_train)
        
        # Evaluate
        accuracy = self.models['category_predictor'].score(X_test_scaled, y_test)
        
        print(f"✅ Category predictor trained - Accuracy: {accuracy:.3f}")
        
        return {
            'model': self.models['category_predictor'],
            'scaler': self.scalers['category_predictor'],
            'encoder': self.encoders['category_target'],
            'features': available_cols,
            'accuracy': accuracy
        }
    
    def train_spending_trend_model(self, df):
        """Train model to identify spending trends"""
        print("📈 Training spending trend model...")
        
        # Create daily spending aggregations
        df['Date'] = df['DateTime'].dt.date
        daily_spending = df[df['Amount'] < 0].groupby('Date').agg({
            'Amount': lambda x: abs(x.sum()),
            'AbsAmount': 'count'
        }).rename(columns={'AbsAmount': 'TransactionCount'})
        
        # Create features for trend analysis
        daily_spending = daily_spending.reset_index()
        daily_spending['Date'] = pd.to_datetime(daily_spending['Date'])
        daily_spending = daily_spending.sort_values('Date')
        
        # Time-based features
        daily_spending['DayOfYear'] = daily_spending['Date'].dt.dayofyear
        daily_spending['DayOfWeek'] = daily_spending['Date'].dt.dayofweek
        daily_spending['Month'] = daily_spending['Date'].dt.month
        daily_spending['IsWeekend'] = (daily_spending['DayOfWeek'] >= 5).astype(int)
        
        # Rolling averages
        daily_spending['MA_7'] = daily_spending['Amount'].rolling(window=7).mean()
        daily_spending['MA_30'] = daily_spending['Amount'].rolling(window=30).mean()
        
        # Prepare features for trend prediction
        feature_cols = ['DayOfYear', 'DayOfWeek', 'Month', 'IsWeekend', 'TransactionCount']
        X = daily_spending[feature_cols].fillna(0)
        y = daily_spending['Amount']
        
        # Remove first 30 days (for rolling average calculation)
        X = X.iloc[30:]
        y = y.iloc[30:]
        
        if len(X) < 10:
            print("⚠️ Not enough data for trend model")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scalers['trend_model'] = StandardScaler()
        X_train_scaled = self.scalers['trend_model'].fit_transform(X_train)
        X_test_scaled = self.scalers['trend_model'].transform(X_test)
        
        # Train model
        self.models['trend_model'] = RandomForestRegressor(n_estimators=50, random_state=42)
        self.models['trend_model'].fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.models['trend_model'].predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Trend model trained - RMSE: ${rmse:.2f}, R²: {r2:.3f}")
        
        return {
            'model': self.models['trend_model'],
            'scaler': self.scalers['trend_model'],
            'features': feature_cols,
            'daily_data': daily_spending,
            'rmse': rmse,
            'r2': r2
        }
    
    def predict_next_month_expenses(self, df):
        """Predict expenses for next month"""
        if 'expense_predictor' not in self.models:
            return None
        
        # Get current month data
        current_month = df['DateTime'].dt.to_period('M').max()
        current_data = df[df['DateTime'].dt.to_period('M') == current_month]
        
        # Create features for next month
        next_month_features = {
            'Month': (current_month + 1).month,
            'Year': (current_month + 1).year,
            'TotalTransactions': len(current_data),  # Use current as baseline
            'AvgTransactionSize': current_data['AbsAmount'].mean(),
            'WeekendTransactions': len(current_data[current_data['IsWeekend'] == 1]),
        }
        
        # Category-wise expenses (use current month as baseline)
        for category in df['Category'].unique():
            cat_expense = abs(current_data[
                (current_data['Category'] == category) & (current_data['Amount'] < 0)
            ]['Amount'].sum())
            next_month_features[f'Expense_{category.replace(" ", "_").replace("&", "and")}'] = cat_expense
        
        # Convert to DataFrame
        features_df = pd.DataFrame([next_month_features])
        
        # Scale features
        features_scaled = self.scalers['expense_predictor'].transform(features_df)
        
        # Make prediction
        prediction = self.models['expense_predictor'].predict(features_scaled)[0]
        
        return {
            'predicted_amount': prediction,
            'current_month_spending': abs(current_data[current_data['Amount'] < 0]['Amount'].sum()),
            'prediction_confidence': 'High' if len(df) > 100 else 'Medium'
        }
    
    def get_spending_insights(self, df):
        """Generate spending insights using trained models"""
        insights = {}
        
        # Monthly expense prediction
        expense_prediction = self.predict_next_month_expenses(df)
        if expense_prediction:
            insights['next_month_prediction'] = expense_prediction
        
        # Anomaly detection on recent transactions
        if 'anomaly_detector' in self.models:
            recent_transactions = df.tail(50)  # Last 50 transactions
            X, _ = self.prepare_features(recent_transactions)
            
            amount_features = pd.DataFrame({
                'Amount': recent_transactions['Amount'],
                'AbsAmount': recent_transactions['AbsAmount'],
                'AmountZScore': (recent_transactions['AbsAmount'] - recent_transactions['AbsAmount'].mean()) / recent_transactions['AbsAmount'].std()
            })
            
            X_anomaly = pd.concat([X, amount_features], axis=1).fillna(0)
            X_scaled = self.scalers['anomaly_detector'].transform(X_anomaly)
            
            anomaly_scores = self.models['anomaly_detector'].decision_function(X_scaled)
            anomalies = self.models['anomaly_detector'].predict(X_scaled)
            
            recent_anomalies = recent_transactions[anomalies == -1]
            insights['recent_anomalies'] = {
                'count': len(recent_anomalies),
                'transactions': recent_anomalies[['DateTime', 'Description', 'Amount']].to_dict('records')
            }
        
        # Spending trends
        if 'trend_model' in self.models:
            # Calculate trend over last 30 days
            last_30_days = df[df['DateTime'] >= (df['DateTime'].max() - timedelta(days=30))]
            daily_spending = last_30_days[last_30_days['Amount'] < 0].groupby(
                last_30_days['DateTime'].dt.date
            )['Amount'].sum().abs()
            
            if len(daily_spending) > 7:
                trend_slope = np.polyfit(range(len(daily_spending)), daily_spending.values, 1)[0]
                insights['spending_trend'] = {
                    'direction': 'increasing' if trend_slope > 0 else 'decreasing',
                    'daily_change': abs(trend_slope),
                    'trend_strength': 'strong' if abs(trend_slope) > 10 else 'moderate'
                }
        
        return insights
    
    def save_models(self, model_dir='models'):
        """Save trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f'{model_dir}/{name}.joblib')
        
        # Save encoders and scalers
        joblib.dump(self.encoders, f'{model_dir}/encoders.joblib')
        joblib.dump(self.scalers, f'{model_dir}/scalers.joblib')
        
        print(f"✅ Models saved to {model_dir}/")
    
    def load_models(self, model_dir='models'):
        """Load saved models"""
        try:
            # Load models
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib') and f not in ['encoders.joblib', 'scalers.joblib']]
            
            for model_file in model_files:
                model_name = model_file.replace('.joblib', '')
                self.models[model_name] = joblib.load(f'{model_dir}/{model_file}')
            
            # Load encoders and scalers
            self.encoders = joblib.load(f'{model_dir}/encoders.joblib')
            self.scalers = joblib.load(f'{model_dir}/scalers.joblib')
            
            print(f"✅ Models loaded from {model_dir}/")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

def main():
    """Train all models"""
    print("🤖 Starting ML model training...")
    
    ml_models = FinancialMLModels()
    
    # Load data
    df = ml_models.load_data()
    print(f"📊 Loaded {len(df)} transactions")
    
    # Train models
    expense_model = ml_models.train_expense_predictor(df)
    anomaly_model = ml_models.train_anomaly_detector(df)
    category_model = ml_models.train_category_predictor(df)
    trend_model = ml_models.train_spending_trend_model(df)
    
    # Generate insights
    print("\n🔮 Generating insights...")
    insights = ml_models.get_spending_insights(df)
    
    # Display insights
    if 'next_month_prediction' in insights:
        pred = insights['next_month_prediction']
        print(f"💰 Next month predicted spending: ${pred['predicted_amount']:.2f}")
        print(f"📊 Current month spending: ${pred['current_month_spending']:.2f}")
    
    if 'recent_anomalies' in insights:
        anomalies = insights['recent_anomalies']
        print(f"🚨 Recent anomalies detected: {anomalies['count']}")
    
    if 'spending_trend' in insights:
        trend = insights['spending_trend']
        print(f"📈 Spending trend: {trend['direction']} (${trend['daily_change']:.2f}/day)")
    
    # Save models
    ml_models.save_models()
    
    print("\n✅ ML model training completed!")

if __name__ == "__main__":
    main()