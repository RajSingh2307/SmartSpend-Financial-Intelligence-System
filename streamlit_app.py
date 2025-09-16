import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import os
import sys

# Add src directory to path
sys.path.append('src')
sys.path.append('.')

try:
    from ml_models import FinancialMLModels
    from data_processor import FinancialDataProcessor
except ImportError:
    st.error("Please make sure ml_models.py and data_processor.py are in the src/ directory or current directory")

# Page config
st.set_page_config(
    page_title="SmartSpend - Financial Intelligence Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SmartSpendDashboard:
    def __init__(self):
        self.ml_models = FinancialMLModels()
        self.processor = FinancialDataProcessor()
        
    def load_data(self):
        """Load processed transaction data"""
        try:
            if os.path.exists('data/processed/transactions.db'):
                conn = sqlite3.connect('data/processed/transactions.db')
                df = pd.read_sql_query("SELECT * FROM transactions", conn)
                conn.close()
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                return df
            elif os.path.exists('data/processed/processed_transactions.csv'):
                df = pd.read_csv('data/processed/processed_transactions.csv')
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                return df
            else:
                return None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def create_overview_metrics(self, df):
        """Create overview metrics cards"""
        # Calculate key metrics
        total_income = df[df['Amount'] > 0]['Amount'].sum()
        total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
        net_flow = total_income - total_expenses
        avg_transaction = df['AbsAmount'].mean()
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💵 Total Income",
                value=f"${total_income:,.2f}",
                delta=f"Last 30 days"
            )
        
        with col2:
            st.metric(
                label="💸 Total Expenses",
                value=f"${total_expenses:,.2f}",
                delta=f"-${total_expenses/12:,.0f}/month avg"
            )
        
        with col3:
            st.metric(
                label="💰 Net Flow",
                value=f"${net_flow:,.2f}",
                delta="Income - Expenses",
                delta_color="normal" if net_flow > 0 else "inverse"
            )
        
        with col4:
            st.metric(
                label="📊 Avg Transaction",
                value=f"${avg_transaction:.2f}",
                delta=f"{len(df)} total transactions"
            )
    
    def create_spending_by_category_chart(self, df):
        """Create spending by category visualization"""
        category_spending = df[df['Amount'] < 0].groupby('Category')['Amount'].sum().abs().sort_values(ascending=True)
        
        fig = px.bar(
            x=category_spending.values,
            y=category_spending.index,
            orientation='h',
            title="💳 Spending by Category",
            labels={'x': 'Amount Spent ($)', 'y': 'Category'},
            color=category_spending.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    def create_monthly_trend_chart(self, df):
        """Create monthly spending trend chart"""
        df['YearMonth'] = df['DateTime'].dt.to_period('M').astype(str)
        
        monthly_data = df.groupby('YearMonth').agg({
            'Amount': lambda x: x[x > 0].sum(),  # Income
            'AbsAmount': lambda x: x[df.loc[x.index, 'Amount'] < 0].sum()  # Expenses
        }).rename(columns={'Amount': 'Income', 'AbsAmount': 'Expenses'})
        
        fig = go.Figure()
        
        # Add income line
        fig.add_trace(go.Scatter(
            x=monthly_data.index,
            y=monthly_data['Income'],
            mode='lines+markers',
            name='Income',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        # Add expenses line
        fig.add_trace(go.Scatter(
            x=monthly_data.index,
            y=monthly_data['Expenses'],
            mode='lines+markers',
            name='Expenses',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="📈 Monthly Income vs Expenses Trend",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_spending_heatmap(self, df):
        """Create spending heatmap by day of week and hour"""
        # Filter expenses only
        expenses = df[df['Amount'] < 0].copy()
        expenses['DayName'] = expenses['DateTime'].dt.strftime('%A')
        expenses['Hour'] = expenses['DateTime'].dt.hour
        
        # Create pivot table
        heatmap_data = expenses.groupby(['DayName', 'Hour'])['AbsAmount'].sum().unstack(fill_value=0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig = px.imshow(
            heatmap_data,
            aspect="auto",
            title="🕒 Spending Patterns: Day of Week vs Hour",
            labels=dict(x="Hour of Day", y="Day of Week", color="Amount Spent ($)"),
            color_continuous_scale="Reds"
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_anomaly_detection_chart(self, df):
        """Create anomaly detection visualization"""
        # Simple anomaly detection based on amount
        Q1 = df['AbsAmount'].quantile(0.25)
        Q3 = df['AbsAmount'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 2.5 * IQR
        
        df['IsAnomaly'] = df['AbsAmount'] > upper_bound
        
        fig = px.scatter(
            df.tail(200),  # Last 200 transactions
            x='DateTime',
            y='AbsAmount',
            color='IsAnomaly',
            title="🚨 Transaction Anomaly Detection",
            labels={'AbsAmount': 'Transaction Amount ($)', 'DateTime': 'Date'},
            color_discrete_map={True: 'red', False: 'blue'},
            hover_data=['Description', 'Category']
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_financial_health_score(self, df):
        """Calculate and display financial health score"""
        # Calculate various financial health indicators
        total_income = df[df['Amount'] > 0]['Amount'].sum()
        total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
        
        # Calculate ratios
        savings_rate = (total_income - total_expenses) / total_income if total_income > 0 else 0
        expense_variance = df[df['Amount'] < 0]['Amount'].std()
        
        # Calculate category diversity (lower is better for expenses)
        category_counts = df[df['Amount'] < 0]['Category'].value_counts()
        category_entropy = -(category_counts / category_counts.sum() * np.log(category_counts / category_counts.sum())).sum()
        
        # Scoring components (0-100 scale)
        savings_score = min(100, max(0, savings_rate * 200))  # 50% savings rate = 100 points
        consistency_score = max(0, 100 - (expense_variance / 100))  # Lower variance = higher score
        diversity_score = min(100, category_entropy * 30)  # Balanced spending across categories
        
        # Overall score
        health_score = (savings_score + consistency_score + diversity_score) / 3
        
        return {
            'overall_score': health_score,
            'savings_rate': savings_rate * 100,
            'consistency_score': consistency_score,
            'diversity_score': diversity_score,
            'grade': self._get_grade(health_score)
        }
    
    def _get_grade(self, score):
        """Convert score to letter grade"""
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'
    
    def create_predictions_section(self, df):
        """Create ML predictions section"""
        st.subheader("🔮 AI-Powered Predictions")
        
        try:
            # Load or train models
            if os.path.exists('models'):
                self.ml_models.load_models()
            
            # Generate predictions and insights
            insights = self.ml_models.get_spending_insights(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Next Month Prediction")
                if 'next_month_prediction' in insights:
                    pred = insights['next_month_prediction']
                    current_spending = pred['current_month_spending']
                    predicted_spending = pred['predicted_amount']
                    
                    change = predicted_spending - current_spending
                    change_pct = (change / current_spending) * 100 if current_spending > 0 else 0
                    
                    st.metric(
                        label="Predicted Next Month Spending",
                        value=f"${predicted_spending:,.2f}",
                        delta=f"{change_pct:+.1f}% vs current month"
                    )
                else:
                    st.info("Training prediction model...")
            
            with col2:
                st.markdown("### Spending Trend")
                if 'spending_trend' in insights:
                    trend = insights['spending_trend']
                    direction_emoji = "📈" if trend['direction'] == 'increasing' else "📉"
                    
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>{direction_emoji} Trend: {trend['direction'].title()}</h4>
                        <p>Daily change: ${trend['daily_change']:.2f}</p>
                        <p>Strength: {trend['trend_strength']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Analyzing spending trends...")
            
            # Anomalies section
            if 'recent_anomalies' in insights:
                if insights['recent_anomalies']['count'] > 0:
                    st.markdown("### 🚨 Recent Unusual Transactions")
                    anomaly_df = pd.DataFrame(insights['recent_anomalies']['transactions'])
                    st.dataframe(anomaly_df)
                else:
                    st.success("No unusual transactions detected recently! 👍")
        
        except Exception as e:
            st.warning(f"Predictions temporarily unavailable: {e}")
    
    def create_recommendations(self, df):
        """Create personalized recommendations"""
        st.subheader("💡 Personalized Recommendations")
        
        # Calculate insights for recommendations
        category_spending = df[df['Amount'] < 0].groupby('Category')['Amount'].sum().abs()
        top_categories = category_spending.nlargest(3)
        
        total_expenses = category_spending.sum()
        monthly_expenses = total_expenses / 12
        
        recommendations = []
        
        # High spending categories
        if 'Food & Dining' in top_categories.index:
            food_pct = (top_categories.get('Food & Dining', 0) / total_expenses) * 100
            if food_pct > 15:
                recommendations.append({
                    'type': 'warning',
                    'title': '🍽️ High Food Spending Detected',
                    'message': f'You spend {food_pct:.1f}% of your budget on food & dining. Consider meal planning or cooking more at home.',
                    'potential_savings': top_categories.get('Food & Dining', 0) * 0.2
                })
        
        # Shopping recommendations
        if 'Shopping' in top_categories.index:
            shopping_pct = (top_categories.get('Shopping', 0) / total_expenses) * 100
            if shopping_pct > 20:
                recommendations.append({
                    'type': 'info',
                    'title': '🛍️ Shopping Optimization',
                    'message': f'Shopping represents {shopping_pct:.1f}% of your expenses. Try the 24-hour rule before purchases.',
                    'potential_savings': top_categories.get('Shopping', 0) * 0.15
                })
        
        # Positive reinforcement
        health_score = self.create_financial_health_score(df)
        if health_score['savings_rate'] > 20:
            recommendations.append({
                'type': 'success',
                'title': '💪 Great Saving Habits!',
                'message': f'You\'re saving {health_score["savings_rate"]:.1f}% of your income. Keep up the excellent work!',
                'potential_savings': 0
            })
        
        # Budget recommendations
        if monthly_expenses > 0:
            recommended_emergency_fund = monthly_expenses * 6
            recommendations.append({
                'type': 'info',
                'title': '🏦 Emergency Fund Goal',
                'message': f'Build an emergency fund of ${recommended_emergency_fund:,.2f} (6 months of expenses).',
                'potential_savings': 0
            })
        
        # Display recommendations
        for rec in recommendations:
            if rec['type'] == 'warning':
                st.markdown(f"""
                <div class="warning-box">
                    <h4>{rec['title']}</h4>
                    <p>{rec['message']}</p>
                    {f"<p><strong>💰 Potential monthly savings: ${rec['potential_savings']/12:,.0f}</strong></p>" if rec['potential_savings'] > 0 else ""}
                </div>
                """, unsafe_allow_html=True)
            elif rec['type'] == 'success':
                st.markdown(f"""
                <div class="success-box">
                    <h4>{rec['title']}</h4>
                    <p>{rec['message']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:  # info
                st.markdown(f"""
                <div class="insight-box">
                    <h4>{rec['title']}</h4>
                    <p>{rec['message']}</p>
                    {f"<p><strong>💰 Potential monthly savings: ${rec['potential_savings']/12:,.0f}</strong></p>" if rec['potential_savings'] > 0 else ""}
                </div>
                """, unsafe_allow_html=True)

def main():
    # Title and description
    st.markdown('<h1 class="main-header">💰 SmartSpend Financial Intelligence</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Transform your financial data into actionable insights with AI-powered analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = SmartSpendDashboard()
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Sample Data", "Upload CSV", "Connect Database"]
    )
    
    df = None
    
    if data_source == "Sample Data":
        # Load sample data
        df = dashboard.load_data()
        if df is None:
            st.error("No sample data found! Please run the data generation script first.")
            st.code("python generate_sample_data.py")
            st.stop()
    
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your transaction CSV",
            type=['csv'],
            help="CSV should contain columns: Date, Description, Category, Amount"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Basic processing
                df = dashboard.processor.clean_data(df)
                df = dashboard.processor.feature_engineering(df)
                st.sidebar.success(f"✅ Loaded {len(df)} transactions")
            except Exception as e:
                st.sidebar.error(f"Error processing file: {e}")
        else:
            st.info("👆 Please upload a CSV file to get started")
            st.stop()
    
    else:  # Connect Database
        st.info("Database connection feature coming soon!")
        st.stop()
    
    # Date range filter
    if df is not None and len(df) > 0:
        min_date = df['DateTime'].min().date()
        max_date = df['DateTime'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['DateTime'].dt.date >= start_date) & 
                   (df['DateTime'].dt.date <= end_date)]
    
    # Category filter
    if df is not None and 'Category' in df.columns:
        categories = ['All'] + sorted(df['Category'].unique().tolist())
        selected_categories = st.sidebar.multiselect(
            "Filter by Category",
            categories,
            default=['All']
        )
        
        if 'All' not in selected_categories:
            df = df[df['Category'].isin(selected_categories)]
    
    # Main dashboard
    if df is not None and len(df) > 0:
        # Overview metrics
        dashboard.create_overview_metrics(df)
        
        st.markdown("---")
        
        # Financial Health Score
        st.subheader("🎯 Financial Health Score")
        health_score = dashboard.create_financial_health_score(df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = health_score['overall_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Grade: {health_score['grade']}"},
                delta = {'reference': 75},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.metric("💰 Savings Rate", f"{health_score['savings_rate']:.1f}%")
            st.metric("📊 Spending Consistency", f"{health_score['consistency_score']:.0f}/100")
        
        with col3:
            st.metric("🎯 Category Balance", f"{health_score['diversity_score']:.0f}/100")
            total_transactions = len(df)
            st.metric("📈 Total Transactions", f"{total_transactions:,}")
        
        with col4:
            avg_daily_spending = abs(df[df['Amount'] < 0]['Amount'].sum()) / len(df['DateTime'].dt.date.unique())
            st.metric("📅 Avg Daily Spending", f"${avg_daily_spending:.2f}")
            
            anomaly_count = (df['AbsAmount'] > df['AbsAmount'].quantile(0.95)).sum()
            st.metric("🚨 Unusual Transactions", f"{anomaly_count}")
        
        st.markdown("---")
        
        # Charts section
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🔍 Analysis", "🔮 Predictions"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_category = dashboard.create_spending_by_category_chart(df)
                st.plotly_chart(fig_category, use_container_width=True)
            
            with col2:
                fig_monthly = dashboard.create_monthly_trend_chart(df)
                st.plotly_chart(fig_monthly, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_heatmap = dashboard.create_spending_heatmap(df)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with col2:
                # Daily spending trend
                daily_spending = df[df['Amount'] < 0].groupby(df['DateTime'].dt.date)['Amount'].sum().abs()
                fig_daily = px.line(
                    x=daily_spending.index,
                    y=daily_spending.values,
                    title="📅 Daily Spending Trend",
                    labels={'x': 'Date', 'y': 'Amount Spent ($)'}
                )
                fig_daily.update_layout(height=400)
                st.plotly_chart(fig_daily, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_anomaly = dashboard.create_anomaly_detection_chart(df)
                st.plotly_chart(fig_anomaly, use_container_width=True)
            
            with col2:
                # Top merchants
                top_merchants = df.groupby('Description')['AbsAmount'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(10)
                fig_merchants = px.bar(
                    x=top_merchants['sum'],
                    y=top_merchants.index,
                    orientation='h',
                    title="🏪 Top Merchants by Spending",
                    labels={'x': 'Total Spent ($)', 'y': 'Merchant'}
                )
                fig_merchants.update_layout(height=400)
                st.plotly_chart(fig_merchants, use_container_width=True)
        
        with tab4:
            dashboard.create_predictions_section(df)
        
        st.markdown("---")
        
        # Recommendations
        dashboard.create_recommendations(df)
        
        st.markdown("---")
        
        # Raw data table
        st.subheader("📋 Transaction Details")
        
        # Search and filter
        search_term = st.text_input("🔍 Search transactions", placeholder="Search by description, category, or amount...")
        
        display_df = df.copy()
        if search_term:
            display_df = display_df[
                display_df['Description'].str.contains(search_term, case=False, na=False) |
                display_df['Category'].str.contains(search_term, case=False, na=False) |
                display_df['Amount'].astype(str).str.contains(search_term, na=False)
            ]
        
        # Display columns selection
        display_columns = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=['DateTime', 'Description', 'Category', 'Amount', 'Account']
        )
        
        if display_columns:
            st.dataframe(
                display_df[display_columns].sort_values('DateTime', ascending=False),
                use_container_width=True,
                height=400
            )
        
        # Export functionality
        st.subheader("📤 Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"smartspend_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export Summary Report"):
                # Create summary report
                health_score = dashboard.create_financial_health_score(df)
                summary_data = {
                    'Metric': [
                        'Total Income', 'Total Expenses', 'Net Flow', 'Financial Health Score',
                        'Savings Rate', 'Average Transaction', 'Most Frequent Category'
                    ],
                    'Value': [
                        f"${df[df['Amount'] > 0]['Amount'].sum():,.2f}",
                        f"${abs(df[df['Amount'] < 0]['Amount'].sum()):,.2f}",
                        f"${df['Amount'].sum():,.2f}",
                        f"{health_score['overall_score']:.1f}/100 ({health_score['grade']})",
                        f"{health_score['savings_rate']:.1f}%",
                        f"${df['AbsAmount'].mean():.2f}",
                        df['Category'].mode().iloc[0] if len(df) > 0 else 'N/A'
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Report",
                    data=csv,
                    file_name=f"smartspend_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            st.info("💡 More export options coming soon!")
        
    else:
        st.warning("No data available. Please check your data source.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚀 SmartSpend Financial Intelligence Dashboard</p>
        <p>Built with Streamlit • Powered by Machine Learning • Made for Financial Freedom</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()