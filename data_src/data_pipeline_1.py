"""
Data Transformation Pipeline - Iteration 1
Basic Statistical Cleaning Pipeline
"""

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import os

class DataPipeline:
    """
    Basic Statistical Cleaning Pipeline - A multi-stage data transformation pipeline
    that handles basic cleaning, statistical analysis, and visualization generation
    """
    
    def __init__(self, config=None):
        """Initialize pipeline with configuration"""
        self.config = config or {
            'date_format': '%Y-%m-%d',
            'region_mapping': {
                'north america': 'North America',
                'north America': 'North America',
                'europe': 'Europe',
                'asiaa': 'Asia',
                'Unknown': 'Unknown'
            },
            'output_dir': '../data_output'
        }
        self.results = {}
        self.metrics = {}
        
    def load_data(self, path):
        """Load data from source"""
        print(f"Loading data from {path}")
        try:
            data = pd.read_csv(path)
            print(f"Successfully loaded {len(data)} records")
            self.metrics['raw_record_count'] = len(data)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
    def clean_data(self, data):
        """Clean and prepare data using basic statistical methods"""
        print("Cleaning data...")
        
        # Make a copy to avoid modifying original
        cleaned = data.copy()
        
        # Track cleaning metrics
        cleaning_metrics = {
            'missing_values_filled': 0,
            'date_format_fixed': 0,
            'region_standardized': 0,
            'invalid_values_fixed': 0
        }
        
        # 1. Fix date formatting
        for i, date in enumerate(cleaned['date']):
            try:
                pd.to_datetime(date)
            except:
                # Try alternative format
                try:
                    fixed_date = pd.to_datetime(date, format='%Y/%m/%d').strftime('%Y-%m-%d')
                    cleaned.at[i, 'date'] = fixed_date
                    cleaning_metrics['date_format_fixed'] += 1
                except:
                    pass
                    
        # 2. Handle missing customer_id with incremental IDs starting from max+1
        max_id = cleaned['customer_id'].dropna().astype(float).max()
        missing_id_mask = cleaned['customer_id'].isna()
        cleaning_metrics['missing_values_filled'] += missing_id_mask.sum()
        
        id_counter = max_id + 1
        for i, is_missing in enumerate(missing_id_mask):
            if is_missing:
                cleaned.at[i, 'customer_id'] = id_counter
                id_counter += 1
        
        # 3. Fix quantity issues
        # Convert non-numeric to NaN
        cleaned['quantity'] = pd.to_numeric(cleaned['quantity'], errors='coerce')
        
        # Replace negative values with absolute value (assuming they're returns)
        neg_mask = cleaned['quantity'] < 0
        cleaned.loc[neg_mask, 'quantity'] = cleaned.loc[neg_mask, 'quantity'].abs()
        cleaning_metrics['invalid_values_fixed'] += neg_mask.sum()
        
        # Fill missing with median value
        median_qty = cleaned['quantity'].median()
        qty_missing_mask = cleaned['quantity'].isna()
        cleaned['quantity'].fillna(median_qty, inplace=True)
        cleaning_metrics['missing_values_filled'] += qty_missing_mask.sum()
        
        # 4. Standardize region names
        for i, region in enumerate(cleaned['customer_region']):
            if pd.notna(region) and region in self.config['region_mapping']:
                cleaned.at[i, 'customer_region'] = self.config['region_mapping'][region]
                cleaning_metrics['region_standardized'] += 1
                
        # Fill missing regions with "Unknown"
        region_missing_mask = cleaned['customer_region'].isna()
        cleaned['customer_region'].fillna('Unknown', inplace=True)
        cleaning_metrics['missing_values_filled'] += region_missing_mask.sum()
        
        # 5. Handle missing prices with product average
        product_avg_prices = cleaned.groupby('product')['price'].transform('mean')
        price_missing_mask = cleaned['price'].isna()
        cleaned['price'].fillna(product_avg_prices, inplace=True)
        cleaning_metrics['missing_values_filled'] += price_missing_mask.sum()
        
        # Store metrics
        self.metrics['cleaning'] = cleaning_metrics
        print(f"Cleaning complete. Fixed {sum(cleaning_metrics.values())} issues.")
        
        return cleaned
    
    def transform_data(self, data):
        """Apply transformations to data"""
        print("Transforming data...")
        
        transformed = data.copy()
        
        # 1. Convert date to datetime
        transformed['date'] = pd.to_datetime(transformed['date'])
        
        # 2. Create month and day features
        transformed['month'] = transformed['date'].dt.month
        transformed['day_of_week'] = transformed['date'].dt.dayofweek
        
        # 3. Calculate total value column
        transformed['total_value'] = transformed['quantity'] * transformed['price']
        
        # 4. Create customer category based on spending
        # Start with a basic categorization
        transformed['customer_category'] = pd.qcut(
            transformed.groupby('customer_id')['total_value'].transform('sum'),
            q=3,
            labels=['Budget', 'Regular', 'Premium']
        )
        
        print("Transformation complete.")
        return transformed
    
    def analyze_data(self, data):
        """Perform analysis on data"""
        print("Analyzing data...")
        
        analysis_results = {}
        
        # 1. Basic summary statistics
        analysis_results['summary_stats'] = data.describe()
        
        # 2. Product popularity
        product_counts = data.groupby('product')['quantity'].sum().sort_values(ascending=False)
        analysis_results['product_popularity'] = product_counts
        
        # 3. Regional sales analysis
        region_sales = data.groupby('customer_region')['total_value'].sum().sort_values(ascending=False)
        analysis_results['region_sales'] = region_sales
        
        # 4. Day of week patterns
        day_sales = data.groupby('day_of_week')['total_value'].sum()
        analysis_results['day_sales_pattern'] = day_sales
        
        # 5. Customer category distribution
        category_counts = data['customer_category'].value_counts()
        analysis_results['customer_categories'] = category_counts
        
        self.results = analysis_results
        print("Analysis complete.")
        return analysis_results
    
    def visualize_results(self, output_dir=None):
        """Generate visualizations from analysis"""
        if output_dir is None:
            output_dir = self.config['output_dir']
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating visualizations in {output_dir}...")
        
        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 12))
        
        # 1. Product Popularity Bar Chart
        plt.subplot(2, 2, 1)
        self.results['product_popularity'].plot(kind='bar', color='skyblue')
        plt.title('Product Popularity')
        plt.ylabel('Total Quantity Sold')
        plt.xticks(rotation=45)
        
        # 2. Regional Sales Pie Chart
        plt.subplot(2, 2, 2)
        self.results['region_sales'].plot(kind='pie', autopct='%1.1f%%')
        plt.title('Sales by Region')
        plt.ylabel('')
        
        # 3. Day of Week Sales Line Chart
        plt.subplot(2, 2, 3)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        plt.plot(days, self.results['day_sales_pattern'].values, marker='o')
        plt.title('Sales by Day of Week')
        plt.xlabel('Day')
        plt.ylabel('Total Sales Value')
        plt.xticks(rotation=45)
        
        # 4. Customer Category Distribution
        plt.subplot(2, 2, 4)
        self.results['customer_categories'].plot(kind='bar', color='lightgreen')
        plt.title('Customer Category Distribution')
        plt.ylabel('Number of Customers')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/analysis_summary.png")
        print(f"Saved visualization to {output_dir}/analysis_summary.png")
        
        plt.close()
        
    def evaluate_performance(self):
        """Evaluate pipeline performance metrics"""
        performance = {
            'data_quality_improvement': {
                'issues_fixed': sum(self.metrics['cleaning'].values()),
                'issues_by_type': self.metrics['cleaning']
            },
            'record_counts': {
                'initial': self.metrics.get('raw_record_count', 0),
                'processed': len(self.results.get('summary_stats', pd.DataFrame()))
            }
        }
        
        print("\nPipeline Performance Metrics:")
        print(f"Total issues fixed: {performance['data_quality_improvement']['issues_fixed']}")
        for issue_type, count in self.metrics['cleaning'].items():
            print(f"  - {issue_type}: {count}")
        print(f"Records processed: {performance['record_counts']['processed']}")
        
        return performance
        
    def run_pipeline(self, data_path):
        """Execute full pipeline"""
        print("\n" + "="*50)
        print("STARTING DATA PIPELINE - ITERATION 1")
        print("="*50 + "\n")
        
        # Step 1: Load data
        data = self.load_data(data_path)
        if data is None:
            print("Pipeline failed: Could not load data")
            return False
            
        # Step 2: Clean data
        cleaned_data = self.clean_data(data)
        
        # Step 3: Transform data
        transformed_data = self.transform_data(cleaned_data)
        
        # Step 4: Analyze data
        self.analyze_data(transformed_data)
        
        # Step 5: Visualize results
        self.visualize_results()
        
        # Step 6: Evaluate performance
        self.evaluate_performance()
        
        print("\n" + "="*50)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*50 + "\n")
        
        return True

# Example usage
if __name__ == "__main__":
    print("Data Pipeline - Iteration 1: Basic Statistical Cleaning")
    import os
    print(f"Current working directory: {os.getcwd()}")
    print(f"Sample data path exists: {os.path.exists('../sample_data.csv')}")
    
    try:
        pipeline = DataPipeline()
        print(f"Pipeline initialized with output directory: {pipeline.config['output_dir']}")
        print(f"Output directory path: {os.path.abspath(pipeline.config['output_dir'])}")
        
        # Ensure output directory exists
        os.makedirs(pipeline.config['output_dir'], exist_ok=True)
        print(f"Output directory created: {os.path.exists(pipeline.config['output_dir'])}")
        
        result = pipeline.run_pipeline("../sample_data.csv")
        print(f"Pipeline execution complete with result: {result}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
