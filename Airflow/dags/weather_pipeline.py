from datetime import datetime, timedelta
import pandas as pd
import requests
import json
import os
from airflow import DAG

# Import PythonOperator (compatible with both old and new versions)
try:
    from airflow.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python_operator import PythonOperator

# Default arguments for the DAG
default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'start_date': datetime(2025, 7, 1),  # Fixed date instead of days_ago
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'daily_weather_pipeline',
    default_args=default_args,
    description='Daily Weather Data ETL Pipeline',
    schedule='0 8 * * *',  # Daily at 8 AM UTC (updated for Airflow 2.4+)
    catchup=False,
    tags=['weather', 'etl', 'daily'],
)

# City coordinates for API calls
CITIES = {
    'Paris': {'latitude': 48.85, 'longitude': 2.35},
    'London': {'latitude': 51.51, 'longitude': -0.13},
    'Berlin': {'latitude': 52.52, 'longitude': 13.41}
}


def extract_weather_data(**context):
    """Extract weather data from Open-Meteo API for all cities"""
    weather_data = []

    for city, coords in CITIES.items():
        try:
            # Build API URL
            url = f"https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': coords['latitude'],
                'longitude': coords['longitude'],
                'current_weather': 'true'
            }

            print(f"Fetching weather data for {city}...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            current_weather = data.get('current_weather', {})

            # Extract required fields
            city_data = {
                'city': city,
                'temperature': current_weather.get('temperature'),
                'windspeed': current_weather.get('windspeed'),
                'weather_code': current_weather.get('weathercode'),
                'timestamp': current_weather.get('time')
            }

            weather_data.append(city_data)
            print(f"Successfully extracted data for {city}: {city_data}")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {city}: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error for {city}: {str(e)}")
            raise

    # Store data for next task (using XCom)
    return weather_data


def transform_weather_data(**context):
    """Transform the extracted weather data"""
    # Get data from previous task
    weather_data = context['task_instance'].xcom_pull(task_ids='extract_weather')

    if not weather_data:
        raise ValueError("No weather data received from extract task")

    print(f"Transforming data for {len(weather_data)} cities...")

    # Create DataFrame
    df = pd.DataFrame(weather_data)

    # Add processing timestamp
    df['extracted_at'] = datetime.utcnow().isoformat()

    # Ensure data types
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df['windspeed'] = pd.to_numeric(df['windspeed'], errors='coerce')
    df['weather_code'] = pd.to_numeric(df['weather_code'], errors='coerce')

    # Reorder columns
    df = df[['city', 'timestamp', 'temperature', 'windspeed', 'weather_code', 'extracted_at']]

    print(f"Transformed DataFrame shape: {df.shape}")
    print(f"DataFrame preview:\n{df.head()}")

    # Convert to dict for XCom (JSON serializable)
    return df.to_dict('records')


def load_weather_data(**context):
    """Load transformed data to CSV file with idempotency"""
    # Get data from transform task
    transformed_data = context['task_instance'].xcom_pull(task_ids='transform_weather')

    if not transformed_data:
        raise ValueError("No transformed data received")

    # Convert back to DataFrame
    df_new = pd.DataFrame(transformed_data)

    # Define the output file path
    output_file = '/opt/airflow/data/weather_data.csv'

    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Handle idempotency - avoid duplicate rows
    if os.path.exists(output_file):
        print("Existing CSV found, checking for duplicates...")
        df_existing = pd.read_csv(output_file)

        # Create unique identifier for each row
        df_new['row_id'] = df_new['city'] + '_' + df_new['timestamp'].astype(str)
        df_existing['row_id'] = df_existing['city'] + '_' + df_existing['timestamp'].astype(str)

        # Filter out duplicates
        df_to_append = df_new[~df_new['row_id'].isin(df_existing['row_id'])]
        df_to_append = df_to_append.drop('row_id', axis=1)

        if df_to_append.empty:
            print("No new data to append - all rows already exist")
            return "No new data added"

        # Append new data
        df_to_append.to_csv(output_file, mode='a', header=False, index=False)
        print(f"Appended {len(df_to_append)} new rows to existing CSV")

    else:
        print("Creating new CSV file...")
        df_new.to_csv(output_file, index=False)
        print(f"Created new CSV with {len(df_new)} rows")

    # Verify the file
    final_df = pd.read_csv(output_file)
    print(f"Final CSV contains {len(final_df)} total rows")

    return f"Successfully loaded data to {output_file}"


# Define tasks
extract_task = PythonOperator(
    task_id='extract_weather',
    python_callable=extract_weather_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_weather',
    python_callable=transform_weather_data,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_weather',
    python_callable=load_weather_data,
    dag=dag,
)

# Set task dependencies
extract_task >> transform_task >> load_task