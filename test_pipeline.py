"""
Simple test script to verify data pipeline functionality
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Print working directory info
    print(f"Current working directory: {os.getcwd()}")
    data_path = "sample_data.csv"
    print(f"Sample data exists: {os.path.exists(data_path)}")
    
    # Create output directory
    output_dir = "pipeline_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at: {os.path.abspath(output_dir)}")
    
    # Load and analyze data
    try:
        print("Loading data...")
        data = pd.read_csv(data_path)
        print(f"Successfully loaded {len(data)} records")
        
        # Simple data cleaning
        print("Cleaning data...")
        data['quantity'] = pd.to_numeric(data['quantity'], errors='coerce')
        data['price'] = pd.to_numeric(data['price'], errors='coerce')
        data['total'] = data['quantity'] * data['price']
        
        # Create a simple visualization
        print("Generating visualization...")
        plt.figure(figsize=(10, 6))
        
        # Product sales chart
        product_sales = data.groupby('product')['quantity'].sum().sort_values(ascending=False)
        product_sales.plot(kind='bar', color='skyblue')
        plt.title('Product Popularity')
        plt.ylabel('Total Quantity Sold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(output_dir, "product_sales.png")
        plt.savefig(output_file)
        print(f"Visualization saved to: {os.path.abspath(output_file)}")
        plt.close()
        
        print("Test completed successfully!")
        return True
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*50)
    print("TESTING DATA PIPELINE FUNCTIONALITY")
    print("="*50)
    main()
