#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Catalog Reconciliation with LLM Analysis Runner
----------------------------------------------
This script runs the catalog reconciliation process and then
performs LLM-powered analysis on the results.
"""

import os
import sys
import logging
import time
import argparse
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_output/reconciliation_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_reconciliation_with_llm_analysis(
    shopify_file: str, 
    sql_file: str, 
    output_dir: str = "data_output",
    api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Run catalog reconciliation process followed by LLM analysis.
    
    Args:
        shopify_file: Path to Shopify products export CSV
        sql_file: Path to SQL database products export (Excel)
        output_dir: Directory for output files
        api_key: Optional API key for the LLM service (defaults to env var)
        
    Returns:
        Dictionary with results summary
    """
    start_time = time.time()
    logger.info(f"Starting reconciliation process with LLM analysis")
    
    # Step 1: Run catalog reconciliation pipeline
    try:
        from data_src.catalog_reconciliation_1 import ProductCatalogReconciliation
        
        # Initialize reconciliation pipeline
        pipeline = ProductCatalogReconciliation(
            shopify_path=shopify_file,
            sql_path=sql_file,
            output_dir=output_dir
        )
        
        # Run reconciliation process
        logger.info("Running catalog reconciliation pipeline")
        
        # First load and preprocess data
        logger.info("Loading data from source files")
        pipeline.load_data()
        
        logger.info("Preprocessing data")
        pipeline.preprocess_data()
        
        # Then run full reconciliation
        logger.info("Performing reconciliation")
        reconciliation_results = pipeline.reconcile_catalog()
        
        logger.info(f"Reconciliation complete in {time.time() - start_time:.2f} seconds")
        logger.info(f"Reconciled {reconciliation_results.get('total_products', 0)} products")
        
    except Exception as e:
        logger.error(f"Error during catalog reconciliation: {str(e)}")
        raise
    
    # Step 2: Run LLM analysis on reconciliation output
    try:
        from data_src.llm_analyzer import run_llm_analysis
        
        logger.info("Running LLM analysis on reconciliation output")
        analysis_results = run_llm_analysis(api_key=api_key, output_dir=output_dir)
        
        logger.info(f"LLM analysis complete")
        
    except Exception as e:
        logger.error(f"Error during LLM analysis: {str(e)}")
        raise
    
    # Step 3: Return combined results
    total_time = time.time() - start_time
    logger.info(f"Total processing complete in {total_time:.2f} seconds")
    
    return {
        "reconciliation_results": reconciliation_results,
        "llm_analysis": {
            "status": "completed" if analysis_results else "failed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "runtime_seconds": total_time
    }

def main():
    """Command line interface for running reconciliation with LLM analysis."""
    parser = argparse.ArgumentParser(
        description="Run catalog reconciliation with LLM analysis"
    )
    
    parser.add_argument(
        "--shopify", 
        type=str, 
        default="files/Shopify_products_export_24_07_2025.csv",
        help="Path to Shopify products export CSV"
    )
    
    parser.add_argument(
        "--sql", 
        type=str, 
        default="files/ITEMS_Supermu_SQL.xlsx", 
        help="Path to SQL database products export (Excel)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data_output",
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="API key for the LLM service (defaults to ANTHROPIC_API_KEY env var)"
    )
    
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Run in mock mode without actual LLM calls"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy SQL file to avoid permission errors (as seen in previous runs)
    from shutil import copyfile
    sql_copy_path = os.path.join(args.output_dir, os.path.basename(args.sql).replace(".xlsx", "_copy.xlsx"))
    try:
        copyfile(args.sql, sql_copy_path)
        logger.info(f"Copied SQL file to {sql_copy_path}")
        sql_file = sql_copy_path
    except Exception as e:
        logger.warning(f"Could not copy SQL file: {str(e)}. Using original file.")
        sql_file = args.sql
    
    # Run the process
    try:
        results = run_reconciliation_with_llm_analysis(
            shopify_file=args.shopify,
            sql_file=sql_file,
            output_dir=args.output_dir,
            api_key=None if args.mock_llm else args.api_key
        )
        
        logger.info("Process completed successfully")
        logger.info(f"Results: {results}")
        
        # Print a message about where to find the output
        print("\nProcess completed successfully!")
        print(f"Output files are in: {os.path.abspath(args.output_dir)}")
        print("Key output files:")
        print(f" - Reconciled catalog: {os.path.join(args.output_dir, 'reconciled_catalog.csv')}")
        print(f" - LLM analysis report: {os.path.join(args.output_dir, 'llm_analysis_report.md')}")
        print(f" - Detailed analysis data: {os.path.join(args.output_dir, 'llm_analysis_results.json')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in reconciliation process: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
