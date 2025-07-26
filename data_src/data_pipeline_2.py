"""
Data Transformation Pipeline - Iteration 2
Machine Learning Enhanced Cleaning Pipeline
"""

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, impute, cluster
from sklearn.ensemble import IsolationForest
import os

class DataPipeline:
    """
    ML Enhanced Cleaning Pipeline - A multi-stage data transformation pipeline
    that uses machine learning techniques for cleaning, anomaly detection,
    and advanced analysis with clustering-based insights
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
            'output_dir': 'output',
            'anomaly_contamination': 0.1,  # Expected proportion of anomalies
            'n_clusters': 3  # Number of customer segments to create
        }
        self.results = {}
        self.metrics = {}
        self.models = {}
        
    def load_data(self, path):
        """Load data from source with validation"""
        print(f"Loading data from {path}")
        try:
            data = pd.read_csv(path)
            # Perform basic validation checks
            expected_columns = ['date', 'customer_id', 'product', 'quantity', 
                               'price', 'customer_region', 'notes']
            missing_cols = set(expected_columns) - set(data.columns)
            
            if missing_cols:
                print(f"Warning: Missing expected columns: {missing_cols}")
                
            print(f"Successfully loaded {len(data)} records")
            self.metrics['raw_record_count'] = len(data)
            self.metrics['raw_column_count'] = len(data.columns)
            
            # Basic data profiling
            self.metrics['missing_values_by_column'] = data.isna().sum().to_dict()
            self.metrics['unique_values_by_column'] = {col: data[col].nunique() for col in data.columns}
            
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
    def clean_data(self, data):
        """Clean and prepare data using advanced ML techniques"""
        print("Cleaning data with ML techniques...")
        
        # Make a copy to avoid modifying original
        cleaned = data.copy()
        
        # Track cleaning metrics
        cleaning_metrics = {
            'missing_values_filled': 0,
            'date_format_fixed': 0,
            'region_standardized': 0,
            'anomalies_detected': 0,
            'outliers_handled': 0
        }
        
        # 1. Fix date formatting with more robust handling
        cleaned['date'] = pd.to_datetime(cleaned['date'], errors='coerce')
        date_missing_mask = cleaned['date'].isna()
        cleaning_metrics['date_format_fixed'] = date_missing_mask.sum()
        
        # Fill missing dates with the most common date
        if date_missing_mask.sum() > 0:
            most_common_date = cleaned['date'].mode()[0]
            cleaned.loc[date_missing_mask, 'date'] = most_common_date
            cleaning_metrics['missing_values_filled'] += date_missing_mask.sum()
                    
        # 2. Handle missing customer_id using KNN imputation for categorical data
        # Convert to numeric for imputation
        cleaned['customer_id'] = pd.to_numeric(cleaned['customer_id'], errors='coerce')
        
        # 3. Fix quantity and price issues
        # Convert to numeric first
        cleaned['quantity'] = pd.to_numeric(cleaned['quantity'], errors='coerce')
        cleaned['price'] = pd.to_numeric(cleaned['price'], errors='coerce')
        
        # 4. Use KNN imputation for missing numeric values
        numeric_cols = ['customer_id', 'quantity', 'price']
        num_missing = cleaned[numeric_cols].isna().sum().sum()
        
        if num_missing > 0:
            # Create temporary columns for imputation
            imp_df = cleaned[numeric_cols].copy()
            
            # Use KNN imputer from scikit-learn
            imputer = impute.KNNImputer(n_neighbors=3)
            imp_values = imputer.fit_transform(imp_df)
            
            # Replace values in original dataframe
            for i, col in enumerate(numeric_cols):
                mask = cleaned[col].isna()
                cleaned.loc[mask, col] = imp_values[mask.values, i]
                cleaning_metrics['missing_values_filled'] += mask.sum()
        
        # 5. Detect outliers using Isolation Forest
        outlier_features = ['quantity', 'price']
        if all(col in cleaned.columns for col in outlier_features):
            outlier_df = cleaned[outlier_features].copy()
            
            # Train isolation forest
            iso_forest = IsolationForest(
                contamination=self.config['anomaly_contamination'],
                random_state=42
            )
            outliers = iso_forest.fit_predict(outlier_df)
            
            # Mark outliers (-1 are outliers, 1 are inliers)
            cleaned['is_outlier'] = outliers == -1
            cleaning_metrics['anomalies_detected'] = sum(cleaned['is_outlier'])
            
            # Store model for later use
            self.models['outlier_detector'] = iso_forest
            
            # Handle outliers - cap at percentiles instead of removing
            for col in outlier_features:
                lower_bound = cleaned[col].quantile(0.01)
                upper_bound = cleaned[col].quantile(0.99)
                
                # Cap outliers
                outlier_mask = cleaned['is_outlier'] & ((cleaned[col] < lower_bound) | (cleaned[col] > upper_bound))
                cleaned.loc[outlier_mask & (cleaned[col] < lower_bound), col] = lower_bound
                cleaned.loc[outlier_mask & (cleaned[col] > upper_bound), col] = upper_bound
                
                cleaning_metrics['outliers_handled'] += outlier_mask.sum()
        
        # 6. Standardize region names with fuzzy matching
        # First standardize known mappings
        for i, region in enumerate(cleaned['customer_region']):
            if pd.notna(region) and region in self.config['region_mapping']:
                cleaned.at[i, 'customer_region'] = self.config['region_mapping'][region]
                cleaning_metrics['region_standardized'] += 1
        
        # Fill missing regions with most frequent value
        region_missing_mask = cleaned['customer_region'].isna()
        if region_missing_mask.sum() > 0:
            most_common_region = cleaned['customer_region'].mode()[0]
            cleaned.loc[region_missing_mask, 'customer_region'] = most_common_region
            cleaning_metrics['missing_values_filled'] += region_missing_mask.sum()
        
        # Store metrics
        self.metrics['cleaning'] = cleaning_metrics
        print(f"ML-based cleaning complete. Fixed {sum(cleaning_metrics.values())} issues.")
        
        return cleaned
    
    def transform_data(self, data):
        """Apply advanced transformations to data"""
        print("Transforming data with advanced features...")
        
        transformed = data.copy()
        
        # 1. Convert date to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(transformed['date']):
            transformed['date'] = pd.to_datetime(transformed['date'], errors='coerce')
        
        # 2. Create enhanced temporal features
        transformed['month'] = transformed['date'].dt.month
        transformed['day_of_week'] = transformed['date'].dt.dayofweek
        transformed['is_weekend'] = transformed['day_of_week'].isin([5, 6]).astype(int)
        transformed['week_of_year'] = transformed['date'].dt.isocalendar().week
        
        # 3. Calculate transaction metrics
        transformed['total_value'] = transformed['quantity'] * transformed['price']
        
        # 4. Create customer level features
        customer_stats = transformed.groupby('customer_id').agg(
            total_spent=('total_value', 'sum'),
            avg_item_price=('price', 'mean'),
            total_items=('quantity', 'sum'),
            purchase_count=('date', 'count')
        ).reset_index()
        
        # Join back to main dataframe
        transformed = pd.merge(
            transformed, 
            customer_stats, 
            on='customer_id', 
            how='left',
            suffixes=('', '_customer')
        )
        
        # 5. Create product level features
        product_stats = transformed.groupby('product').agg(
            avg_price=('price', 'mean'),
            total_sold=('quantity', 'sum')
        ).reset_index()
        
        # Join back to main dataframe
        transformed = pd.merge(
            transformed,
            product_stats,
            on='product',
            how='left',
            suffixes=('', '_product')
        )
        
        # 6. Advanced feature: price deviation from product average
        transformed['price_deviation'] = transformed['price'] - transformed['avg_price_product']
        transformed['price_deviation_pct'] = (transformed['price_deviation'] / transformed['avg_price_product']) * 100
        
        print("Advanced transformation complete.")
        return transformed
    
    def analyze_data(self, data):
        """Perform ML-enhanced analysis on data"""
        print("Analyzing data with machine learning techniques...")
        
        analysis_results = {}
        
        # 1. Basic summary statistics
        analysis_results['summary_stats'] = data.describe()
        
        # 2. Product popularity and trends
        product_analysis = data.groupby(['product', 'month']).agg(
            total_quantity=('quantity', 'sum'),
            total_value=('total_value', 'sum'),
            avg_price=('price', 'mean')
        ).reset_index()
        
        analysis_results['product_trends'] = product_analysis
        
        # 3. Regional sales analysis with temporal patterns
        region_time_analysis = data.groupby(['customer_region', 'day_of_week']).agg(
            total_value=('total_value', 'sum')
        ).reset_index().pivot(index='customer_region', columns='day_of_week', values='total_value')
        
        analysis_results['region_time_patterns'] = region_time_analysis
        
        # 4. Customer segmentation using K-Means clustering
        # Select features for clustering
        if all(col in data.columns for col in ['total_spent', 'avg_item_price', 'total_items', 'purchase_count']):
            cluster_features = ['total_spent', 'avg_item_price', 'total_items', 'purchase_count']
            cluster_data = data.groupby('customer_id')[cluster_features].mean().reset_index()
            
            # Normalize data for clustering
            scaler = preprocessing.StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data[cluster_features])
            
            # Apply K-Means
            kmeans = cluster.KMeans(n_clusters=self.config['n_clusters'], random_state=42)
            cluster_data['segment'] = kmeans.fit_predict(scaled_data)
            
            # Store model
            self.models['customer_segmentation'] = kmeans
            
            # Calculate segment profiles
            segment_profiles = cluster_data.groupby('segment')[cluster_features].mean()
            
            # Interpret segments
            segment_names = []
            for segment in range(self.config['n_clusters']):
                profile = segment_profiles.loc[segment]
                
                if profile['total_spent'] > segment_profiles['total_spent'].median():
                    if profile['purchase_count'] > segment_profiles['purchase_count'].median():
                        name = "High-Value Frequent"
                    else:
                        name = "Big Spenders"
                else:
                    if profile['purchase_count'] > segment_profiles['purchase_count'].median():
                        name = "Frequent Budget"
                    else:
                        name = "Occasional Shoppers"
                        
                segment_names.append(name)
            
            # Add segment names to profiles
            segment_profiles['segment_name'] = segment_names
            
            # Join segment info back to customer data
            customer_segments = cluster_data[['customer_id', 'segment']]
            segment_name_map = {i: name for i, name in enumerate(segment_names)}
            customer_segments['segment_name'] = customer_segments['segment'].map(segment_name_map)
            
            analysis_results['customer_segments'] = customer_segments
            analysis_results['segment_profiles'] = segment_profiles
        
        # 5. Correlation analysis
        if all(col in data.columns for col in ['total_value', 'is_weekend', 'price_deviation_pct']):
            corr_features = ['total_value', 'quantity', 'price', 'is_weekend', 
                            'price_deviation_pct', 'total_items', 'purchase_count']
            corr_features = [f for f in corr_features if f in data.columns]
            
            correlation_matrix = data[corr_features].corr()
            analysis_results['correlation_matrix'] = correlation_matrix
        
        self.results = analysis_results
        print("ML-enhanced analysis complete.")
        return analysis_results
    
    def visualize_results(self, output_dir=None):
        """Generate advanced visualizations from analysis"""
        if output_dir is None:
            output_dir = self.config['output_dir']
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating advanced visualizations in {output_dir}...")
        
        # Set the style
        sns.set(style="whitegrid")
        
        # 1. Product Trends Over Time
        if 'product_trends' in self.results:
            plt.figure(figsize=(12, 6))
            
            # Filter to top products for clarity
            top_products = self.results['product_trends'].groupby('product')['total_value'].sum().nlargest(3).index
            filtered_data = self.results['product_trends'][self.results['product_trends']['product'].isin(top_products)]
            
            # Plot trends
            for product, group in filtered_data.groupby('product'):
                plt.plot(group['month'], group['total_value'], marker='o', label=product)
                
            plt.title('Top Product Sales Trends by Month', fontsize=15)
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Total Sales Value', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/product_trends.png")
            plt.close()
        
        # 2. Customer Segment Analysis
        if 'customer_segments' in self.results and 'segment_profiles' in self.results:
            plt.figure(figsize=(10, 8))
            
            segment_counts = self.results['customer_segments']['segment_name'].value_counts()
            segment_profiles = self.results['segment_profiles'].reset_index()
            
            # Create a radar chart for segment profiles
            categories = ['total_spent', 'avg_item_price', 'total_items', 'purchase_count']
            categories = [c for c in categories if c in segment_profiles.columns]
            
            if categories:
                # Normalize the values for radar chart
                segment_profiles_norm = segment_profiles.copy()
                for cat in categories:
                    segment_profiles_norm[cat] = (segment_profiles_norm[cat] - segment_profiles_norm[cat].min()) / \
                                              (segment_profiles_norm[cat].max() - segment_profiles_norm[cat].min())
                
                # Create the radar chart
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]  # close the loop
                
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                
                for i, row in segment_profiles_norm.iterrows():
                    values = row[categories].tolist()
                    values += values[:1]  # close the loop
                    ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['segment_name'])
                    ax.fill(angles, values, alpha=0.1)
                
                # Set category labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels([c.replace('_', ' ').title() for c in categories])
                ax.set_title('Customer Segment Profiles', fontsize=15)
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/segment_profiles.png")
                plt.close()
            
            # Create a pie chart for segment distribution
            plt.figure(figsize=(10, 7))
            plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', 
                   shadow=True, startangle=140)
            plt.title('Customer Segment Distribution', fontsize=15)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/segment_distribution.png")
            plt.close()
        
        # 3. Regional Sales Patterns by Day of Week
        if 'region_time_patterns' in self.results:
            plt.figure(figsize=(12, 8))
            
            # Create a heatmap
            data = self.results['region_time_patterns']
            sns.heatmap(data, cmap='YlGnBu', annot=True, fmt='.0f')
            
            # Customize the plot
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            plt.xticks(ticks=np.arange(0.5, len(days)), labels=days, rotation=45)
            plt.yticks(rotation=0)
            plt.title('Regional Sales by Day of Week', fontsize=15)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/region_day_sales.png")
            plt.close()
        
        # 4. Correlation Matrix Visualization
        if 'correlation_matrix' in self.results:
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(self.results['correlation_matrix']))
            sns.heatmap(self.results['correlation_matrix'], mask=mask, cmap='coolwarm', 
                       vmax=1, vmin=-1, center=0, annot=True, fmt='.2f')
            plt.title('Feature Correlation Matrix', fontsize=15)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_matrix.png")
            plt.close()
        
        print(f"Advanced visualizations saved to {output_dir}/")
        
    def evaluate_performance(self):
        """Evaluate pipeline performance with detailed metrics"""
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
        
        # Calculate analytical value metrics
        if 'customer_segments' in self.results:
            performance['analytical_value'] = {
                'segments_identified': len(self.results['segment_profiles']),
                'segment_distribution': self.results['customer_segments']['segment_name'].value_counts().to_dict()
            }
        
        # Calculate model performance if applicable
        if 'outlier_detector' in self.models:
            performance['model_metrics'] = {
                'outlier_detection': {
                    'contamination_parameter': self.config['anomaly_contamination'],
                    'outliers_detected': int(self.metrics['cleaning'].get('anomalies_detected', 0))
                }
            }
        
        print("\nML Pipeline Performance Metrics:")
        print(f"Total issues fixed: {performance['data_quality_improvement']['issues_fixed']}")
        for issue_type, count in self.metrics['cleaning'].items():
            print(f"  - {issue_type}: {count}")
        print(f"Records processed: {performance['record_counts']['processed']}")
        
        if 'analytical_value' in performance:
            print(f"Customer segments identified: {performance['analytical_value']['segments_identified']}")
            print("Segment distribution:")
            for segment, count in performance['analytical_value']['segment_distribution'].items():
                print(f"  - {segment}: {count}")
        
        return performance
        
    def run_pipeline(self, data_path):
        """Execute full ML-enhanced pipeline"""
        print("\n" + "="*50)
        print("STARTING ML-ENHANCED DATA PIPELINE - ITERATION 2")
        print("="*50 + "\n")
        
        # Step 1: Load data
        data = self.load_data(data_path)
        if data is None:
            print("Pipeline failed: Could not load data")
            return False
            
        # Step 2: Clean data with ML methods
        cleaned_data = self.clean_data(data)
        
        # Step 3: Transform data with advanced features
        transformed_data = self.transform_data(cleaned_data)
        
        # Step 4: Analyze data with ML techniques
        self.analyze_data(transformed_data)
        
        # Step 5: Generate advanced visualizations
        self.visualize_results()
        
        # Step 6: Evaluate performance with detailed metrics
        self.evaluate_performance()
        
        print("\n" + "="*50)
        print("ML-ENHANCED PIPELINE EXECUTION COMPLETE")
        print("="*50 + "\n")
        
        return True

# Example usage
if __name__ == "__main__":
    print("Data Pipeline - Iteration 2: Machine Learning Enhanced Cleaning")
    pipeline = DataPipeline()
    pipeline.run_pipeline("../sample_data.csv")
