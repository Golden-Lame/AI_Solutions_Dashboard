import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from web_logs_db import init_logs_db, insert_many_logs
from auth_db import init_db, register_user, authenticate_user
import calendar
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from scipy import stats

# Function for predictive product demand analysis
def predict_product_demand(df):
    """
    Predicts future product demand using a linear trend based on the last 7 days.
    Limits to the last 200 days of data and top 3 products for performance.
    Returns predictions and model accuracy metrics (R² and RMSE) for each product.
    """
    df_recent = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=200))]
    daily_product_demand = df_recent.groupby(['date', 'product_name']).size().reset_index(name='demand')
    products = daily_product_demand['product_name'].unique()
    prediction_results = {}
    accuracy_metrics = {}
    future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=14, freq='D').date
    
    for product in products[:3]:  # Limit to top 3 products
        product_data = daily_product_demand[daily_product_demand['product_name'] == product].sort_values('date')
        print(f"Product: {product}, Total rows: {len(product_data)}")  # Debug output
        if len(product_data) < 7:
            print(f"Skipping {product} due to insufficient data (<7 days)")
            continue
        recent_data = product_data.tail(7)  # Use last 7 days
        print(f"Recent data rows for {product}: {len(recent_data)}")  # Debug output
        if len(recent_data) < 2:
            base_demand = recent_data['demand'].iloc[-1]
            prediction_results[product] = [base_demand] * 14
            accuracy_metrics[product] = {'R2': None, 'RMSE': None}  # No model trained
        else:
            X = np.arange(len(recent_data)).reshape(-1, 1)
            y = recent_data['demand'].values
            variance = np.var(y)
            print(f"Variance for {product}: {variance}")  # Debug variance
            if variance == 0:
                base_demand = np.mean(y)
                prediction_results[product] = [base_demand] * 14
                accuracy_metrics[product] = {'R2': 0.0, 'RMSE': 0.0}  # Handle zero variance
            else:
                model = LinearRegression()
                model.fit(X, y)
                future_X = np.arange(len(recent_data), len(recent_data) + 14).reshape(-1, 1)
                predictions = model.predict(future_X)
                predictions = [max(0, round(p)) for p in predictions]
                prediction_results[product] = predictions
                # Calculate accuracy metrics
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                rmse = sqrt(mean_squared_error(y, y_pred))
                accuracy_metrics[product] = {'R2': r2, 'RMSE': rmse}
                print(f"R2 for {product}: {r2}, RMSE for {product}: {rmse}")  # Debug metrics
    
    prediction_df = pd.DataFrame({
        'date': list(future_dates) * len(prediction_results),
        'product_name': [p for p in prediction_results.keys() for _ in range(len(future_dates))],
        'predicted_demand': [val for product_vals in prediction_results.values() for val in product_vals]
    })
    return prediction_df, accuracy_metrics

# Function to visualize product demand predictions
def create_demand_forecast_chart(prediction_df):
    """
    Creates a line chart for the 14-day product demand forecast with confidence bounds.
    """
    if prediction_df.empty:
        return None
    fig = go.Figure()
    for product in prediction_df['product_name'].unique():
        product_data = prediction_df[prediction_df['product_name'] == product]
        fig.add_trace(go.Scatter(
            x=product_data['date'],
            y=product_data['predicted_demand'],
            mode='lines+markers',
            name=f'{product}',
            line=dict(width=2)
        ))
        # Note: upper_bound and lower_bound are missing in prediction_df; add them in predict_product_demand if needed
        if 'upper_bound' in product_data.columns and 'lower_bound' in product_data.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([product_data['date'], product_data['date'][::-1]]),
                y=pd.concat([product_data['upper_bound'], product_data['lower_bound'][::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    fig.update_layout(
        title='14-Day Product Demand Forecast (Top 3 Products)',
        xaxis_title='Date',
        yaxis_title='Predicted Demand',
        legend_title='Product',
        hovermode='x unified',
        height=250
    )
    return fig

# Function to predict and compare actual vs predicted sales revenue
def predict_actual_vs_revenue(df):
    """
    Predicts revenue for the next 14 days and compares with historical data.
    Limits to the last 90 days for performance.
    Returns predictions and model accuracy metrics (R² and RMSE).
    """
    df_recent = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=90))]
    daily_revenue = df_recent.groupby('date')['revenue_generated'].sum().reset_index()
    
    print(f"Total revenue days: {len(daily_revenue)}")  # Debug output
    if len(daily_revenue) < 7:
        print("Warning: Insufficient data (<7 days) for revenue prediction. Using zeros.")
        future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=14, freq='D').date
        return pd.DataFrame({
            'date': list(daily_revenue['date']) + list(future_dates),
            'actual_revenue': list(daily_revenue['revenue_generated']) + [0.0] * 14,
            'predicted_revenue': [0.0] * len(daily_revenue) + [0.0] * 14,
            'lower_bound': [0.0] * (len(daily_revenue) + 14),
            'upper_bound': [0.0] * (len(daily_revenue) + 14)
        }), {'R2': None, 'RMSE': None}
    
    recent_data = daily_revenue.tail(7)  # Use last 7 days
    print(f"Recent revenue data rows: {len(recent_data)}")  # Debug output
    X = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data['revenue_generated'].values
    variance = np.var(y)
    print(f"Revenue variance: {variance}")  # Debug variance
    if variance == 0:
        base_revenue = np.mean(y)
        all_predictions = [base_revenue] * (len(daily_revenue) + 14)
        accuracy_metrics = {'R2': 0.0, 'RMSE': 0.0}  # Handle zero variance
    else:
        model = LinearRegression()
        model.fit(X, y)
        historical_X = np.arange(len(daily_revenue)).reshape(-1, 1)
        historical_predictions = model.predict(historical_X)
        future_X = np.arange(len(daily_revenue), len(daily_revenue) + 14).reshape(-1, 1)
        future_predictions = model.predict(future_X)
        all_predictions = np.concatenate([historical_predictions, future_predictions])
        # Calculate accuracy metrics
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = sqrt(mean_squared_error(y, y_pred))
        accuracy_metrics = {'R2': r2, 'RMSE': rmse}
        print(f"R2 for revenue: {r2}, RMSE for revenue: {rmse}")  # Debug metrics
    
    std_dev = np.std(y) if len(y) > 1 else 10.0
    lower_bound = all_predictions - std_dev
    upper_bound = all_predictions + std_dev
    lower_bound = np.maximum(lower_bound, 0)
    
    all_dates = list(daily_revenue['date']) + list(pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=14, freq='D').date)
    result_df = pd.DataFrame({
        'date': all_dates,
        'actual_revenue': list(daily_revenue['revenue_generated']) + [0.0] * 14,
        'predicted_revenue': all_predictions,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })
    
    return result_df, accuracy_metrics

# Function to visualize actual vs predicted sales revenue
def create_actual_vs_revenue_chart(prediction_df):
    """
    Creates a line chart comparing actual and predicted revenue with confidence bounds.
    """
    if prediction_df.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prediction_df['date'],
        y=prediction_df['actual_revenue'],
        mode='markers+lines',
        name='Actual Revenue',
        line=dict(width=2, color='#ff7f0e', dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=prediction_df['date'],
        y=prediction_df['predicted_revenue'],
        mode='lines+markers',
        name='Predicted Revenue',
        line=dict(width=2, color='#1f77b4')
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([prediction_df['date'], prediction_df['date'][::-1]]),
        y=pd.concat([prediction_df['upper_bound'], prediction_df['lower_bound'][::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Sales Revenue (Last 90 Days + 14-Day Forecast)',
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        legend_title='Legend',
        hovermode='x unified',
        height=250,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def generate_alerts(df):
    """
    Generates user notifications/alerts based on key metrics in the filtered data.
    Returns a list of alert messages.
    """
    alerts = []

    # Calculate conversion rate
    total_visits = len(df)
    total_conversions = len(df[df['conversion'] == 'Yes'])
    conversion_rate = (total_conversions / total_visits * 100) if total_visits > 0 else 0
    if conversion_rate < 5 and total_visits > 0:
        alerts.append(f"⚠️ **Low Conversion Rate Alert**: Conversion rate is {conversion_rate:.2f}% (below 5%). Consider optimizing the user journey or marketing campaigns.")

    # Calculate bounce rate (sessions with duration < 30 seconds)
    bounce_sessions = len(df[df['duration_on_site'] < 30])
    bounce_rate = (bounce_sessions / total_visits * 100) if total_visits > 0 else 0
    if bounce_rate > 50 and total_visits > 0:
        alerts.append(f"⚠️ **High Bounce Rate Alert**: {bounce_rate:.2f}% of sessions have a duration < 30 seconds. Review landing page content or user engagement strategies.")

    # Calculate revenue change (month-over-month)
    if 'date' in df.columns:
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_revenue = df.groupby('month')['revenue_generated'].sum().reset_index()
        if len(monthly_revenue) >= 2:
            current_revenue = monthly_revenue.iloc[-1]['revenue_generated']
            previous_revenue = monthly_revenue.iloc[-2]['revenue_generated']
            revenue_change = ((current_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue > 0 else 0
            if revenue_change < -20:
                alerts.append(f"⚠️ **Revenue Drop Alert**: Revenue dropped by {abs(revenue_change):.2f}% compared to the previous month. Investigate potential causes.")

    return alerts

def fetch_logs_data():
    """
    Fetches and cleans data from web_logs.db, handling duplicates and missing values.
    Adds derived features for analysis.
    """
    conn = sqlite3.connect("web_logs.db")
    query = """
    SELECT 
        timestamp, 
        ip_address, 
        country, 
        user_id, 
        product_name, 
        request_type, 
        demo_requested, 
        promo_event_interested, 
        ai_assistant_used, 
        duration_on_site, 
        pages_visited, 
        conversion, 
        revenue_generated, 
        browser,
        sales_agent
    FROM web_logs
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        st.sidebar.warning(f"Removed {duplicates_removed} duplicate rows.")

    # Handle missing values using .loc
    missing_values = df.isnull().sum()
    if missing_values.any():
        for column, default in {
            'duration_on_site': 0,
            'pages_visited': 1,
            'revenue_generated': 0,
            'conversion': 'No',
            'sales_agent': 'Unknown'
        }.items():
            df.loc[df[column].isnull(), column] = default
        st.sidebar.warning(f"Handled missing values: {missing_values[missing_values > 0]} replaced with defaults.")

    # Derive features using .loc where applicable
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['month_name'] = df['timestamp'].dt.month_name()
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['session_time_per_page'] = df['duration_on_site'] / df['pages_visited'].replace(0, 1)

    return df

# Function to create summary metrics with smaller cards
def display_summary_metrics(df):
    col1, col2, col3, col4 = st.columns(4)
    total_visits = len(df)
    total_conversions = df[df['conversion'] == 'Yes'].shape[0]
    conversion_rate = (total_conversions / total_visits) * 100 if total_visits > 0 else 0
    total_revenue = df['revenue_generated'].sum()
    avg_order = total_revenue / total_conversions if total_conversions > 0 else 0

    with col1:
        st.markdown(
            """
            <div style='background-color: #D1E8FF; padding: 8px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
                <h5 style='margin: 0; color: #1E88E5; font-size: 14px;'>Total Visits</h5>
                <h3 style='margin: 0; color: #0D47A1; font-size: 18px;'>{}</h3>
            </div>
            """.format(f"{total_visits:,}"),
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div style='background-color: #FFD1DC; padding: 8px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
                <h5 style='margin: 0; color: #E91E63; font-size: 14px;'>Conversion Rate</h5>
                <h3 style='margin: 0; color: #AD1457; font-size: 18px;'>{:.2f}%</h3>
            </div>
            """.format(conversion_rate),
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div style='background-color: #D1FFD7; padding: 8px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
                <h5 style='margin: 0; color: #4CAF50; font-size: 14px;'>Total Revenue</h5>
                <h3 style='margin: 0; color: #2E7D32; font-size: 18px;'>{}</h3>
            </div>
            """.format(f"${total_revenue:,.2f}"),
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div style='background-color: #FFF3D1; padding: 8px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
                <h5 style='margin: 0; color: #FFCA28; font-size: 14px;'>Avg. Order Value</h5>
                <h3 style='margin: 0; color: #F57F17; font-size: 18px;'>{:.2f}</h3>
            </div>
            """.format(avg_order),
            unsafe_allow_html=True
        )
    return conversion_rate

# Function to create sales trend chart
def create_sales_trend(df):
    if df.empty:
        return go.Figure().update_layout(title='Daily Revenue and Visitors (No Data)', height=350)
    daily_sales = df.groupby('date').agg({'revenue_generated': 'sum', 'user_id': 'nunique'}).reset_index()
    daily_sales.columns = ['date', 'revenue', 'visitors']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['revenue'], name='Revenue', line=dict(color='#1f77b4', width=3), mode='lines'))
    fig.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['visitors'], name='Visitors', line=dict(color='#ff7f0e', width=2, dash='dot'), mode='lines', yaxis='y2'))
    fig.update_layout(
        title='Daily Revenue and Visitors', xaxis=dict(title='Date'),
        yaxis=dict(title='Revenue ($)', side='left', showgrid=False),
        yaxis2=dict(title='Visitors', side='right', overlaying='y', showgrid=False),
        legend=dict(x=0.01, y=0.99), hovermode='x unified', height=350, margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def create_gauge_chart(value, title, target_value=100000000):
    """
    Creates a gauge chart to assess sales performance with a target.
    Args:
        value (float): The current total revenue in $.
        title (str): The title of the gauge.
        target_value (float): The target revenue in $ (default: $800,000).
    """
    max_range = max(value * 1.5, target_value * 1.5)
    thresholds = [
        {'min': 0, 'max': target_value * 0.5, 'color': '#FF0000', 'label': 'Poor'},  # Red
        {'min': target_value * 0.5, 'max': target_value, 'color': '#FFFF00', 'label': 'Fair'},  # Yellow
        {'min': target_value, 'max': max_range, 'color': '#00FF00', 'label': 'Great'}  # Green
    ]
    
    status = "Unknown"
    for threshold in thresholds:
        if threshold['min'] <= value <= threshold['max']:
            status = threshold['label']
            break
    
    performance_text = f"{status} - {'Above' if value >= target_value else 'Below'} Target (${target_value:,.2f})"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': target_value, 'increasing': {'color': "#2E7D32"}, 'decreasing': {'color': "#D32F2F"}},
        title={'text': f"{title}", 'font': {'size': 20}},
        number={'suffix': ' $', 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, max_range], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "#00008B"},  # Dark Blue
            'steps': [
                {'range': [t['min'], t['max']], 'color': t['color'], 'line': {'color': "black", 'width': 2}} for t in thresholds
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target_value
            }
        }
    ))
    
    for i, threshold in enumerate(thresholds):
        fig.add_annotation(
            x=threshold['min'] + (threshold['max'] - threshold['min']) / 2,
            y=0.3,
            xref='x', yref='paper',
            text=threshold['label'],
            showarrow=False,
            font=dict(size=12, color="black"),
            align='center'
        )
    
    fig.add_annotation(
        text=performance_text,
        xref="paper", yref="paper",
        x=0.5, y=0.1,
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=40),
        font=dict(size=14)
    )
    return fig

def create_sales_performance_chart(df):
    if df.empty or 'sales_agent' not in df.columns:
        return go.Figure().update_layout(title='No Sales Performance Data Available', height=350)

    performance = df.groupby('sales_agent').agg({
        'revenue_generated': 'sum',
        'conversion': lambda x: len(x[x == 'Yes']),
        'user_id': 'count'
    }).reset_index()
    performance.columns = ['Sales Agent', 'Total Revenue ($)', 'Conversions', 'Total Sessions']

    fig = go.Figure()
    fig.add_trace(go.Bar(x=performance['Sales Agent'], y=performance['Total Revenue ($)'], name='Total Revenue ($)', marker_color='#1f77b4'))
    fig.add_trace(go.Bar(x=performance['Sales Agent'], y=performance['Conversions'], name='Conversions', marker_color='#ff7f0e'))
    fig.add_trace(go.Bar(x=performance['Sales Agent'], y=performance['Total Sessions'], name='Total Sessions', marker_color='#2ca02c'))
    fig.update_layout(
        title='Sales Performance by Agent',
        xaxis=dict(title='Sales Agent'),
        yaxis=dict(title='Value'),
        barmode='group',
        height=250,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# Function to create a monthly trend analysis with KPI
def create_monthly_trend(df):
    monthly_data = df.groupby(['month', 'month_name']).agg({
        'user_id': 'count',
        'conversion': lambda x: (x == 'Yes').sum(),
        'revenue_generated': 'sum'
    }).reset_index()
    monthly_data['conversion_rate'] = (monthly_data['conversion'] / monthly_data['user_id']) * 100
    month_order = {month: i for i, month in enumerate(calendar.month_name[1:])}
    monthly_data['month_idx'] = monthly_data['month_name'].map(month_order)
    monthly_data = monthly_data.sort_values('month_idx')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_data['month_name'], y=monthly_data['revenue_generated'], name='Revenue ($)', line=dict(color='#1f77b4', width=2), mode='lines+markers'))
    fig.add_trace(go.Scatter(
        x=monthly_data['month_name'], y=monthly_data['conversion_rate'],
        name='Conversion Rate (%)', mode='lines+markers', marker=dict(size=6, color='#ff7f0e'),
        line=dict(width=1, color='#ff7f0e'), yaxis='y2'
    ))
    if len(monthly_data) > 1:
        conv_change = ((monthly_data['conversion_rate'].iloc[-1] - monthly_data['conversion_rate'].iloc[0]) / monthly_data['conversion_rate'].iloc[0]) * 100 if monthly_data['conversion_rate'].iloc[0] != 0 else 0
        fig.add_annotation(
            x=monthly_data['month_name'].iloc[-1], y=monthly_data['conversion_rate'].iloc[-1],
            text=f"Conv. Change: {conv_change:+.1f}%",
            showarrow=True, arrowhead=2, ax=20, ay=-30, bgcolor="white", bordercolor="black", borderwidth=1
        )
    fig.update_layout(
        title='Monthly Performance Trends', xaxis=dict(title='Month'),
        yaxis=dict(title='Revenue ($)', side='left'),
        yaxis2=dict(title='Conv. Rate (%)', side='right', overlaying='y', range=[0, max(monthly_data['conversion_rate'])*1.2]),
        legend=dict(x=0.01, y=0.99), margin=dict(l=10, r=60, b=40, t=40), height=250
    )
    return fig

# Function to create product performance chart (treemap)
def create_product_performance(df):
    product_metrics = df.groupby('product_name').agg({
        'user_id': 'count',
        'conversion': lambda x: (x == 'Yes').sum(),
        'revenue_generated': 'sum'
    }).reset_index()
    product_metrics['conversion_rate'] = (product_metrics['conversion'] / product_metrics['user_id']) * 100
    product_metrics = product_metrics.sort_values('revenue_generated', ascending=False)
    fig = px.treemap(
        product_metrics,
        path=['product_name'],
        values='revenue_generated',
        color='conversion_rate',
        color_continuous_scale='Viridis',
        title='Product Performance by Revenue and Conversion Rate',
        labels={'revenue_generated': 'Revenue ($)', 'conversion_rate': 'Conversion Rate (%)'},
        height=250
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

# Function to create sales performance by country (bar chart)
def create_time_analysis(df):
    country_data = df.groupby('country').agg({
        'revenue_generated': 'sum',
        'conversion': lambda x: (x == 'Yes').sum() / len(x) * 100
    }).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=country_data['country'],
        y=country_data['revenue_generated'],
        name='Revenue ($)',
        marker_color='rgb(54, 162, 235)'
    ))
    fig.add_trace(go.Scatter(
        x=country_data['country'],
        y=country_data['conversion'],
        name='Conversion Rate (%)',
        mode='lines+markers',
        marker=dict(size=6, color='rgb(255, 99, 132)'),
        line=dict(width=1, color='rgb(255, 99, 132)'),
        yaxis='y2'
    ))
    fig.update_layout(
        title='Sales Performance by Country',
        xaxis=dict(title='Country', tickangle=45),
        yaxis=dict(title='Revenue ($)', side='left', showgrid=False),
        yaxis2=dict(title='Conversion Rate (%)', side='right', overlaying='y', showgrid=False),
        legend=dict(x=0.01, y=0.99),
        height=250,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# Function to create conversion by day of week (donut chart)
def create_conversion_by_dow_chart(df):
    dow_data = df.groupby('day_of_week').agg({
        'user_id': 'count',
        'conversion': lambda x: (x == 'Yes').sum() / len(x) * 100
    }).reset_index()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_data['day_idx'] = dow_data['day_of_week'].apply(lambda x: days_order.index(x))
    dow_data = dow_data.sort_values('day_idx')
    fig = px.pie(
        dow_data,
        names='day_of_week',
        values='conversion',
        hole=0.4,
        title='Conversion Rate by Day of Week',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textinfo='percent+label', textposition='inside')
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig

# Initialize auth DB
init_db()

# Page configuration
st.set_page_config(
    page_title="AI Solutions Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #0D47A1;
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    .chart-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 0.5rem;
    }
    .sidebar-logo {
        width: 100%;
        margin-bottom: 0.5rem;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-height: 90vh;
        overflow-y: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# Authentication sidebar
try:
    st.sidebar.image("Dash logo.jpg", use_container_width=True, caption="Sales Analytics Dashboard", output_format="auto", clamp=False)
except FileNotFoundError:
    st.sidebar.warning("Logo image not found. Please ensure 'Dash logo.jpg' is in the correct directory.")

st.sidebar.title("🔐 Authentication")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    st.sidebar.success("✔ Logged in")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.experimental_rerun()
else:
    auth_mode = st.sidebar.radio("Select Mode", ["Login", "Sign Up"])
    if auth_mode == "Sign Up":
        new_user = st.sidebar.text_input("New Username")
        new_pass = st.sidebar.text_input("New Password", type="password")
        confirm_pass = st.sidebar.text_input("Confirm Password", type="password")
        if st.sidebar.button("Create Account"):
            if new_pass != confirm_pass:
                st.sidebar.error("⚠ Passwords do not match.")
            elif new_user == "" or new_pass == "":
                st.sidebar.error("⚠ Username and password cannot be empty.")
            elif register_user(new_user, new_pass):
                st.sidebar.success("✔ Account created. Please log in.")
            else:
                st.sidebar.error("⚠ Username already exists.")
    else:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.success(f"Welcome, {username}!")
            else:
                st.sidebar.error("❌ Invalid username or password")
    st.warning("🔒 Please log in or sign up to access the dashboard.")
    st.stop()

# Sidebar filters and navigation
st.sidebar.markdown("## 📊 Dashboard Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "Performance Analysis", "AI Insights", "Sales Performance", "Data Explorer"]
)

st.sidebar.markdown("## 📅 Filters")
try:
    df = fetch_logs_data()
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
    else:
        df_filtered = df
    
    all_countries = df['country'].unique().tolist()
    selected_countries = st.sidebar.multiselect("Select Countries", options=all_countries, default=all_countries)
    if selected_countries:
        df_filtered = df_filtered[df_filtered['country'].isin(selected_countries)]
    
    all_products = df['product_name'].unique().tolist()
    selected_products = st.sidebar.multiselect("Select Products", options=all_products, default=all_products)
    if selected_products:
        df_filtered = df_filtered[df_filtered['product_name'].isin(selected_products)]
    
    all_request_types = df['request_type'].unique().tolist()
    selected_request_types = st.sidebar.multiselect("Select Request Types", options=all_request_types, default=all_request_types)
    if selected_request_types:
        df_filtered = df_filtered[df_filtered['request_type'].isin(selected_request_types)]
    
    sales_agents = sorted(df['sales_agent'].dropna().unique())
    selected_agents = st.sidebar.multiselect("Select Sales Agents", sales_agents, default=sales_agents)
    if selected_agents:
        df_filtered = df_filtered[df_filtered['sales_agent'].isin(selected_agents)]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"Last data refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if len(df_filtered) == 0:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        st.stop()

    # Page Rendering
    if page == "Overview":
        st.markdown('<h1 class="main-header">AI Solutions Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Overview</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        total_visits = len(df_filtered)
        total_conversions = df_filtered[df_filtered['conversion'] == 'Yes'].shape[0]
        conversion_rate = (total_conversions / total_visits) * 100 if total_visits > 0 else 0
        total_revenue = df_filtered['revenue_generated'].sum()
        avg_order = total_revenue / total_conversions if total_conversions > 0 else 0

        with col1:
            st.markdown(
                """
                <div style='background-color: #D1E8FF; padding: 8px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
                    <h5 style='margin: 0; color: #1E88E5; font-size: 14px;'>Total Visits</h5>
                    <h3 style='margin: 0; color: #0D47A1; font-size: 18px;'>{:,}</h3>
                </div>
                """.format(total_visits),
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                """
                <div style='background-color: #FFD1DC; padding: 8px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
                    <h5 style='margin: 0; color: #E91E63; font-size: 14px;'>Conversion Rate</h5>
                    <h3 style='margin: 0; color: #AD1457; font-size: 18px;'>{:.2f}%</h3>
                </div>
                """.format(conversion_rate),
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                """
                <div style='background-color: #D1FFD7; padding: 8px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
                    <h5 style='margin: 0; color: #4CAF50; font-size: 14px;'>Total Revenue</h5>
                    <h3 style='margin: 0; color: #2E7D32; font-size: 18px;'>${:,.2f}</h3>
                </div>
                """.format(total_revenue),
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                """
                <div style='background-color: #FFF3D1; padding: 8px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
                    <h5 style='margin: 0; color: #FFCA28; font-size: 14px;'>Avg. Order Value</h5>
                    <h3 style='margin: 0; color: #F57F17; font-size: 18px;'>{:.2f}</h3>
                </div>
                """.format(avg_order),
                unsafe_allow_html=True
            )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            sales_fig = create_sales_trend(df_filtered)
            st.plotly_chart(sales_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            gauge_fig = create_gauge_chart(total_revenue, "Sales Performance")
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Performance Analysis":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            monthly_fig = create_monthly_trend(df_filtered)
            st.plotly_chart(monthly_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            product_fig = create_product_performance(df_filtered)
            st.plotly_chart(product_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            hourly_fig = create_time_analysis(df_filtered)
            st.plotly_chart(hourly_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            dow_fig = create_conversion_by_dow_chart(df_filtered)
            st.plotly_chart(dow_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif page == "AI Insights":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<h4>Product Demand Forecast</h4>', unsafe_allow_html=True)
            with st.spinner("Generating product demand forecast..."):
                df_filtered_recent = df_filtered[df_filtered['date'] >= (df_filtered['date'].max() - pd.Timedelta(days=90))]
                demand_predictions, demand_accuracy = predict_product_demand(df_filtered_recent)
                if not demand_predictions.empty and len(df_filtered_recent) > 0:
                    recent_demand = df_filtered_recent.groupby(['date', 'product_name']).size().reset_index(name='demand')
                    std_dev = recent_demand['demand'].std() if len(recent_demand) > 1 else 10
                    demand_predictions['lower_bound'] = demand_predictions['predicted_demand'] - std_dev
                    demand_predictions['upper_bound'] = demand_predictions['predicted_demand'] + std_dev
                    demand_predictions['lower_bound'] = demand_predictions['lower_bound'].apply(lambda x: max(0, x))
            if not demand_predictions.empty:
                demand_chart = create_demand_forecast_chart(demand_predictions)
                if demand_chart:
                    st.plotly_chart(demand_chart, use_container_width=True)
                with st.expander("View Demand Forecast Data"):
                    st.dataframe(demand_predictions[['date', 'product_name', 'predicted_demand', 'lower_bound', 'upper_bound']], height=150)
                # Display accuracy metrics
                st.markdown('<h5>Model Accuracy Metrics</h5>', unsafe_allow_html=True)
                accuracy_data = [
                    {'Product': product, 'R² Score': metrics['R2'] if metrics['R2'] is not None else 'N/A', 'RMSE': f"{metrics['RMSE']:.2f}" if metrics['RMSE'] is not None else 'N/A'}
                    for product, metrics in demand_accuracy.items()
                ]
                st.table(accuracy_data)
            else:
                st.warning("Not enough data to generate product demand forecasts.")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<h4>Actual vs Predicted Sales Revenue</h4>', unsafe_allow_html=True)
            with st.spinner("Generating revenue analysis..."):
                revenue_predictions, revenue_accuracy = predict_actual_vs_revenue(df_filtered)
            if not revenue_predictions.empty:
                revenue_chart = create_actual_vs_revenue_chart(revenue_predictions)
                if revenue_chart:
                    st.plotly_chart(revenue_chart, use_container_width=True)
                with st.expander("View Revenue Data"):
                    st.dataframe(revenue_predictions[['date', 'actual_revenue', 'predicted_revenue', 'lower_bound', 'upper_bound']], height=150)
                # Display accuracy metrics
                st.markdown('<h5>Model Accuracy Metrics</h5>', unsafe_allow_html=True)
                st.markdown(f"R² Score: {revenue_accuracy['R2']:.3f}" if revenue_accuracy['R2'] is not None else "R² Score: N/A")
                st.markdown(f"RMSE: ${revenue_accuracy['RMSE']:.2f}" if revenue_accuracy['RMSE'] is not None else "RMSE: N/A")
            else:
                st.warning("Not enough data to generate revenue analysis.")
            st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Sales Performance":
        st.markdown('<h2 class="main-header">Sales Performance</h2>', unsafe_allow_html=True)
        st.markdown('<p>Analyze the performance of sales agents based on revenue, conversions, and sessions.</p>', unsafe_allow_html=True)

        if df_filtered.empty:
            st.warning("No data available with the current filters.")
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<h4>Performance by Sales Agent</h4>', unsafe_allow_html=True)
                performance_chart = create_sales_performance_chart(df_filtered)
                st.plotly_chart(performance_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<h4>Performance Metrics</h4>', unsafe_allow_html=True)
                performance_data = df_filtered.groupby('sales_agent').agg({
                    'revenue_generated': 'sum',
                    'conversion': lambda x: len(x[x == 'Yes']),
                    'user_id': 'count'
                }).reset_index()
                performance_data.columns = ['Sales Agent', 'Total Revenue ($)', 'Conversions', 'Total Sessions']
                st.dataframe(performance_data, height=250)
                st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Data Explorer":
        st.markdown('<h2 class="main-header">Data Explorer</h2>', unsafe_allow_html=True)
        st.markdown('<p>Explore the raw data with applied filters and export it for further analysis.</p>', unsafe_allow_html=True)

        # Generate and display alerts
        alerts = generate_alerts(df_filtered)
        if alerts:
            st.markdown('<div class="alert-container">', unsafe_allow_html=True)
            for alert in alerts:
                st.warning(alert)
            st.markdown('</div>', unsafe_allow_html=True)

        # Display the filtered data
        if not df_filtered.empty:
            st.dataframe(df_filtered, height=300)
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="ai_solutions_analytics_data.csv",
                mime="text/csv",
            )
        else:
            st.warning("No data available with the current filters.")

    # Footer
    st.markdown("---")
    st.markdown("AI Sales Solutions Analytics Dashboard | Developed for the Sales Team Members/ Marketers Committee")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure you have run the import_logs.py script to generate sample data!")