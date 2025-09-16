import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import sqlite3
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FinancialDataProcessor:
    def __init__(self):
        self.category_keywords = {
            'Food & Dining': ['restaurant', 'food', 'dining', 'cafe', 'coffee', 'pizza', 'burger', 'mcdonald', 'starbucks', 'kfc'],
            'Groceries': ['grocery', 'supermarket', 'walmart', 'target', 'kroger', 'safeway', 'costco', 'whole foods'],
            'Transportation': ['gas', 'fuel', 'uber', 'lyft', 'taxi', 'metro', 'transit', 'parking', 'shell', 'exxon', 'bp'],
            'Entertainment': ['movie', 'theater', 'netflix', 'spotify', 'game', 'steam', 'concert', 'entertainment'],
            'Shopping': ['amazon', 'ebay', 'shopping', 'store', 'mall', 'nike', 'apple', 'best buy'],
            'Bills & Utilities': ['electric', 'water', 'internet', 'phone', 'insurance', 'utility', 'bill'],
            'Healthcare': ['pharmacy', 'doctor', 'hospital', 'medical', 'dental', 'cvs', 'clinic'],
            'Income': ['salary', 'paycheck', 'deposit', 'income', 'payment', 'bonus', 'refund']
        }
        
        self.le_category = LabelEncoder()
        self.le_merchant = LabelEncoder()
        self.scaler = StandardScaler()
    
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded {len(df)} transactions from {file_path}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean and standardize the data"""
        df_clean = df.copy()
        
        # Handle missing values
        df_clean['Description'] = df_clean['Description'].fillna('Unknown Transaction')
        df_clean['Category'] = df_clean['Category'].fillna('Other')
        
        # Standardize date format
        if 'DateTime' not in df_clean.columns and 'Date' in df_clean.columns:
            if 'Time' in df_clean.columns:
                df_clean['DateTime'] = pd.to_datetime(df_clean['Date'] + ' ' + df_clean['Time'])
            else:
                df_clean['DateTime'] = pd.to_datetime(df_clean['Date'])
        else:
            df_clean['DateTime'] = pd.to_datetime(df_clean['DateTime'])
        
        # Clean description text
        df_clean['Description_Clean'] = df_clean['Description'].apply(self._clean_text)
        
        # Ensure Amount is numeric
        df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce')
        
        # Remove rows with missing amounts
        df_clean = df_clean.dropna(subset=['Amount'])
        
        # Sort by date
        df_clean = df_clean.sort_values('DateTime').reset_index(drop=True)
        
        print(f"✅ Data cleaned: {len(df_clean)} transactions remaining")
        return df_clean
    
    def _clean_text(self, text):
        """Clean text description"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    
    def auto_categorize(self, df):
        """Automatically categorize transactions based on description"""
        df_cat = df.copy()
        
        # Only categorize 'Other' or missing categories
        uncategorized_mask = (df_cat['Category'].isin(['Other', 'Unknown', '']) | 
                             df_cat['Category'].isna())
        
        for idx in df_cat[uncategorized_mask].index:
            description = df_cat.loc[idx, 'Description_Clean']
            predicted_category = self._predict_category(description)
            if predicted_category:
                df_cat.loc[idx, 'Category'] = predicted_category
        
        return df_cat
    
    def _predict_category(self, description):
        """Predict category based on keywords"""
        description = description.lower()
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in description:
                    return category
        
        return 'Other'
    
    def feature_engineering(self, df):
        """Create additional features for analysis and modeling"""
        df_features = df.copy()
        
        # Time-based features
        df_features['Year'] = df_features['DateTime'].dt.year
        df_features['Month'] = df_features['DateTime'].dt.month
        df_features['Day'] = df_features['DateTime'].dt.day
        df_features['DayOfWeek'] = df_features['DateTime'].dt.dayofweek
        df_features['Hour'] = df_features['DateTime'].dt.hour
        df_features['IsWeekend'] = (df_features['DayOfWeek'] >= 5).astype(int)
        
        # Month names for better visualization
        df_features['MonthName'] = df_features['DateTime'].dt.strftime('%B')
        df_features['DayName'] = df_features['DateTime'].dt.strftime('%A')
        
        # Amount-based features
        df_features['AbsAmount'] = abs(df_features['Amount'])
        df_features['IsExpense'] = (df_features['Amount'] < 0).astype(int)
        df_features['IsIncome'] = (df_features['Amount'] > 0).astype(int)
        
        # Rolling statistics (7-day and 30-day windows)
        df_features = df_features.sort_values('DateTime')
        df_features['RollingMean_7d'] = df_features['Amount'].rolling(window=7, min_periods=1).mean()
        df_features['RollingStd_7d'] = df_features['Amount'].rolling(window=7, min_periods=1).std()
        df_features['RollingMean_30d'] = df_features['Amount'].rolling(window=30, min_periods=1).mean()
        
        # Merchant frequency
        merchant_counts = df_features['Description'].value_counts()
        df_features['MerchantFrequency'] = df_features['Description'].map(merchant_counts)
        
        # Category frequency
        category_counts = df_features['Category'].value_counts()
        df_features['CategoryFrequency'] = df_features['Category'].map(category_counts)
        
        # Days since last transaction in same category
        df_features['DaysSinceLastCategoryTransaction'] = (
            df_features.groupby('Category')['DateTime']
            .diff().dt.days.fillna(0)
        )
        
        print(f"✅ Feature engineering completed: {len(df_features.columns)} features")
        return df_features
    
    def detect_anomalies(self, df, method='iqr'):
        """Detect anomalous transactions"""
        df_anomaly = df.copy()
        
        if method == 'iqr':
            # Use IQR method for each category
            df_anomaly['IsAnomaly'] = False
            
            for category in df_anomaly['Category'].unique():
                cat_mask = df_anomaly['Category'] == category
                cat_amounts = df_anomaly.loc[cat_mask, 'AbsAmount']
                
                if len(cat_amounts) < 5:  # Skip categories with too few transactions
                    continue
                
                Q1 = cat_amounts.quantile(0.25)
                Q3 = cat_amounts.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 2.5 * IQR  # More sensitive threshold
                upper_bound = Q3 + 2.5 * IQR
                
                anomaly_mask = (cat_amounts < lower_bound) | (cat_amounts > upper_bound)
                df_anomaly.loc[cat_mask & anomaly_mask, 'IsAnomaly'] = True
        
        anomaly_count = df_anomaly['IsAnomaly'].sum()
        print(f"✅ Anomaly detection completed: {anomaly_count} anomalies found")
        
        return df_anomaly
    
    def calculate_financial_metrics(self, df):
        """Calculate key financial metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['total_transactions'] = len(df)
        metrics['total_expenses'] = abs(df[df['Amount'] < 0]['Amount'].sum())
        metrics['total_income'] = df[df['Amount'] > 0]['Amount'].sum()
        metrics['net_flow'] = metrics['total_income'] - metrics['total_expenses']
        
        # Time-based metrics
        df['YearMonth'] = df['DateTime'].dt.to_period('M')
        monthly_expenses = df[df['Amount'] < 0].groupby('YearMonth')['Amount'].sum().abs()
        monthly_income = df[df['Amount'] > 0].groupby('YearMonth')['Amount'].sum()
        
        metrics['avg_monthly_expenses'] = monthly_expenses.mean()
        metrics['avg_monthly_income'] = monthly_income.mean()
        metrics['expense_volatility'] = monthly_expenses.std()
        
        # Category insights
        category_expenses = df[df['Amount'] < 0].groupby('Category')['Amount'].sum().abs()
        metrics['top_expense_categories'] = category_expenses.nlargest(5).to_dict()
        
        # Behavioral metrics
        metrics['avg_transaction_size'] = df['AbsAmount'].mean()
        metrics['largest_expense'] = df[df['Amount'] < 0]['Amount'].min()
        metrics['most_frequent_merchant'] = df['Description'].value_counts().index[0]
        
        print(f"✅ Financial metrics calculated")
        return metrics
    
    def save_to_database(self, df, db_path='data/processed/transactions.db'):
        """Save processed data to SQLite database"""
        conn = sqlite3.connect(db_path)
        for col in df.columns:
            if pd.api.types.is_period_dtype(df[col]):
                df[col] = df[col].astype(str)
        try:
            df.to_sql('transactions', conn, if_exists='replace', index=False)
            print(f"✅ Data saved to database: {db_path}")
            
            # Create indices for better performance
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_datetime ON transactions(DateTime)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON transactions(Category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_amount ON transactions(Amount)")
            conn.commit()
            
        except Exception as e:
            print(f"❌ Error saving to database: {e}")
        finally:
            conn.close()
    
    def save_processed_data(self, df, file_path='data/processed/processed_transactions.csv'):
        """Save processed data to CSV"""
        try:
            df.to_csv(file_path, index=False)
            print(f"✅ Processed data saved to: {file_path}")
        except Exception as e:
            print(f"❌ Error saving processed data: {e}")

def main():
    """Main processing pipeline"""
    processor = FinancialDataProcessor()
    
    # Load data
    df = processor.load_data('data/sample/sample_transactions.csv')
    if df is None:
        print("❌ Please run generate_sample_data.py first to create sample data")
        return
    
    print("\n" + "="*50)
    print("🔄 PROCESSING PIPELINE STARTED")
    print("="*50)
    
    # Clean data
    df_clean = processor.clean_data(df)
    
    # Auto-categorize
    df_categorized = processor.auto_categorize(df_clean)
    
    # Feature engineering
    df_features = processor.feature_engineering(df_categorized)
    
    # Anomaly detection
    df_final = processor.detect_anomalies(df_features)
    
    # Calculate metrics
    metrics = processor.calculate_financial_metrics(df_final)
    
    # Save processed data
    processor.save_processed_data(df_final)
    processor.save_to_database(df_final)
    
    print("\n" + "="*50)
    print("📊 FINANCIAL SUMMARY")
    print("="*50)
    print(f"💰 Total Income: ${metrics['total_income']:,.2f}")
    print(f"💸 Total Expenses: ${metrics['total_expenses']:,.2f}")
    print(f"💵 Net Flow: ${metrics['net_flow']:,.2f}")
    print(f"📈 Avg Monthly Income: ${metrics['avg_monthly_income']:,.2f}")
    print(f"📉 Avg Monthly Expenses: ${metrics['avg_monthly_expenses']:,.2f}")
    print(f"🎯 Most Frequent Merchant: {metrics['most_frequent_merchant']}")
    
    print(f"\n🏷️ Top Expense Categories:")
    for cat, amount in metrics['top_expense_categories'].items():
        print(f"  • {cat}: ${amount:,.2f}")
    
    print(f"\n🚨 Anomalies Detected: {df_final['IsAnomaly'].sum()}")
    
    print("\n✅ Processing pipeline completed successfully!")

if __name__ == "__main__":
    main()