import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

class FinancialDataGenerator:
    def __init__(self):
        self.categories = {
            'Food & Dining': {
                'merchants': ['McDonald\'s', 'Starbucks', 'Subway', 'Pizza Hut', 'Local Restaurant', 'KFC', 'Domino\'s'],
                'amount_range': (5, 50),
                'frequency': 0.25
            },
            'Groceries': {
                'merchants': ['Walmart', 'Target', 'Kroger', 'Safeway', 'Whole Foods', 'Costco'],
                'amount_range': (30, 200),
                'frequency': 0.15
            },
            'Transportation': {
                'merchants': ['Shell', 'BP', 'Exxon', 'Uber', 'Lyft', 'Metro Transit', 'Parking Meter'],
                'amount_range': (15, 80),
                'frequency': 0.12
            },
            'Entertainment': {
                'merchants': ['Netflix', 'Spotify', 'AMC Theaters', 'Steam', 'PlayStation', 'Concert Venue'],
                'amount_range': (10, 100),
                'frequency': 0.08
            },
            'Shopping': {
                'merchants': ['Amazon', 'eBay', 'Best Buy', 'Macy\'s', 'Nike', 'Apple Store', 'Target'],
                'amount_range': (25, 300),
                'frequency': 0.10
            },
            'Bills & Utilities': {
                'merchants': ['Electric Company', 'Water Dept', 'Internet Provider', 'Phone Company', 'Insurance Co'],
                'amount_range': (50, 250),
                'frequency': 0.05
            },
            'Healthcare': {
                'merchants': ['CVS Pharmacy', 'Dr. Smith Office', 'Hospital', 'Dental Clinic', 'Eye Care Center'],
                'amount_range': (20, 500),
                'frequency': 0.04
            },
            'Income': {
                'merchants': ['Salary Deposit', 'Freelance Payment', 'Investment Return', 'Bonus', 'Tax Refund'],
                'amount_range': (1000, 5000),
                'frequency': 0.05
            }
        }
    
    def generate_transactions(self, start_date, end_date, num_transactions=1000):
        transactions = []
        current_date = start_date
        
        while len(transactions) < num_transactions and current_date <= end_date:
            # Determine if this day should have transactions (weekdays more likely)
            if current_date.weekday() < 5:  # Monday to Friday
                daily_transactions = np.random.poisson(3)  # Average 3 transactions on weekdays
            else:
                daily_transactions = np.random.poisson(1.5)  # Less on weekends
            
            for _ in range(daily_transactions):
                if len(transactions) >= num_transactions:
                    break
                
                # Select category based on frequency
                category = np.random.choice(
                    list(self.categories.keys()),
                    p=[self.categories[cat]['frequency'] / sum(cat_info['frequency'] for cat_info in self.categories.values()) 
                       for cat in self.categories.keys()]
                )
                
                cat_info = self.categories[category]
                
                # Generate transaction details
                merchant = random.choice(cat_info['merchants'])
                
                # Amount (negative for expenses, positive for income)
                if category == 'Income':
                    amount = round(np.random.uniform(*cat_info['amount_range']), 2)
                else:
                    amount = -round(np.random.uniform(*cat_info['amount_range']), 2)
                
                # Add some seasonal variation
                if category == 'Food & Dining' and current_date.month in [11, 12]:  # Holiday season
                    amount *= 1.3
                elif category == 'Shopping' and current_date.month in [11, 12]:
                    amount *= 1.5
                elif category == 'Entertainment' and current_date.month in [6, 7, 8]:  # Summer
                    amount *= 1.2
                
                # Generate transaction time
                transaction_time = current_date + timedelta(
                    hours=random.randint(6, 22),
                    minutes=random.randint(0, 59)
                )
                
                transaction = {
                    'Date': transaction_time.strftime('%Y-%m-%d'),
                    'Time': transaction_time.strftime('%H:%M:%S'),
                    'Description': merchant,
                    'Category': category,
                    'Amount': round(amount, 2),
                    'Account': random.choice(['Checking', 'Credit Card', 'Savings']),
                    'Transaction_ID': fake.uuid4()
                }
                
                transactions.append(transaction)
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(transactions[:num_transactions])
    
    def add_anomalies(self, df, anomaly_rate=0.02):
        """Add some anomalous transactions for testing anomaly detection"""
        num_anomalies = int(len(df) * anomaly_rate)
        anomaly_indices = np.random.choice(df.index, num_anomalies, replace=False)
        
        for idx in anomaly_indices:
            # Make some transactions unusually large
            if df.loc[idx, 'Category'] != 'Income':
                df.loc[idx, 'Amount'] *= np.random.uniform(5, 10)  # 5-10x larger
                df.loc[idx, 'Description'] += ' (UNUSUAL)'
        
        return df

def main():
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/sample', exist_ok=True)
    
    generator = FinancialDataGenerator()
    
    # Generate 12 months of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print("Generating sample financial data...")
    df = generator.generate_transactions(start_date, end_date, num_transactions=1500)
    
    # Add some anomalies
    df = generator.add_anomalies(df)
    
    # Sort by date
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    # Save to CSV
    output_path = 'data/sample/sample_transactions.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df)} transactions")
    print(f"📊 Data saved to: {output_path}")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"💰 Total spent: ${abs(df[df['Amount'] < 0]['Amount'].sum()):,.2f}")
    print(f"💵 Total income: ${df[df['Amount'] > 0]['Amount'].sum():,.2f}")
    
    # Display sample transactions
    print("\n🔍 Sample transactions:")
    print(df[['Date', 'Description', 'Category', 'Amount']].head(10))
    
    # Category breakdown
    print("\n📈 Category breakdown:")
    category_summary = df.groupby('Category')['Amount'].agg(['count', 'sum']).round(2)
    print(category_summary)

if __name__ == "__main__":
    main()