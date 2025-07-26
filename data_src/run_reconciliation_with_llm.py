#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Reconciliation with LLM Analysis Integration
------------------------------------------------
This script runs the complete catalog reconciliation pipeline with integrated
LLM analysis of the output files.

Usage:
    python run_reconciliation_with_llm.py [--api-key YOUR_API_KEY]
"""

import os
import sys
import logging
import argparse
import traceback
from catalog_reconciliation_1 import ProductCatalogReconciliation
from llm_analyzer import LLMAnalyzer, run_llm_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("reconciliation_pipeline")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run catalog reconciliation with LLM analysis")
    parser.add_argument("--api-key", type=str, help="API key for LLM service (defaults to ANTHROPIC_API_KEY env var)")
    parser.add_argument("--output-dir", type=str, default="data_output", help="Directory for output files")
    parser.add_argument("--shopify-path", type=str, default=os.path.join('files', 'Shopify_products_export_24_07_2025.csv'),
                       help="Path to Shopify export CSV file")
    parser.add_argument("--sql-path", type=str, default=os.path.join('data_output', 'ITEMS_Supermu_SQL_copy.xlsx'),
                       help="Path to SQL export Excel file")
    args = parser.parse_args()
    
    # Configure paths
    shopify_path = args.shopify_path
    sql_path = args.sql_path
    output_dir = args.output_dir
    
    logger.info(f"Using files:\n - Shopify: {os.path.abspath(shopify_path)}\n - SQL: {os.path.abspath(sql_path)}")
    logger.info(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Verify files exist
    if not os.path.exists(shopify_path):
        logger.error(f"Shopify file not found: {shopify_path}")
        return
    if not os.path.exists(sql_path):
        logger.error(f"SQL file not found: {sql_path}")
        return
            
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Run the reconciliation pipeline
        logger.info("\n===== RUNNING RECONCILIATION PIPELINE =====")
        
        # Initialize and run pipeline
        pipeline = ProductCatalogReconciliation(shopify_path, sql_path, output_dir)
        
        logger.info("\n1. Loading data from both sources...")
        pipeline.load_data()
        
        logger.info("\n2. Preprocessing data...")
        pipeline.preprocess_data()
        
        logger.info("\n3. Reconciling catalog data...")
        try:
            reconciled_data = pipeline.reconcile_catalog()
            logger.info(f"Reconcile catalog completed successfully with {len(reconciled_data)} products")
        except Exception as e:
            logger.error(f"Error in reconcile_catalog: {str(e)}")
            traceback.print_exc()
            raise
            
        logger.info("\n4. Generating output files and visualizations...")
        try:
            pipeline.generate_output()
            logger.info("Output generation completed successfully")
        except Exception as e:
            logger.error(f"Error in generate_output: {str(e)}")
            traceback.print_exc()
            raise
        
        logger.info("\n5. Evaluating reconciliation quality...")
        try:
            evaluation = pipeline.evaluate_reconciliation()
            logger.info("Evaluation completed successfully")
        except Exception as e:
            logger.error(f"Error in evaluate_reconciliation: {str(e)}")
            traceback.print_exc()
            raise
        
        # Step 2: Run LLM analysis on the output files
        logger.info("\n===== RUNNING LLM ANALYSIS =====")
        try:
            analysis_results = run_llm_analysis(api_key=args.api_key, output_dir=output_dir)
            logger.info("LLM analysis completed successfully")
            
            # Print a summary of the analysis
            print("\n===== LLM ANALYSIS SUMMARY =====")
            if "recommendations" in analysis_results and "summary" in analysis_results["recommendations"]:
                print(analysis_results["recommendations"]["summary"])
                
            # Print path to detailed results
            print(f"\nDetailed analysis results saved to: {os.path.join(output_dir, 'llm_analysis_report.md')}")
        
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            traceback.print_exc()
        
        # Print reconciliation summary
        print("\n===== RECONCILIATION SUMMARY =====")
        print(f"Total products: {evaluation.get('total_products', 'N/A')}")
        print(f"Matched products: {evaluation.get('matched_products', 'N/A')}")
        print(f"Match rate: {evaluation.get('match_rate', 0):.2%}")
        print(f"Average match confidence: {evaluation.get('match_confidence', 0):.2f}")
        print(f"Unmatched products: {evaluation.get('unmatched_shopify', 0)} from Shopify, {evaluation.get('unmatched_sql', 0)} from SQL")
        
        if 'resolution_percentages' in evaluation:
            res = evaluation['resolution_percentages']
            print("\nResolution sources:")
            print(f"  - Shopify: {res.get('shopify', 0):.1f}%")
            print(f"  - SQL: {res.get('sql', 0):.1f}%")
            print(f"  - Combined (identical): {res.get('combined', 0):.1f}%")
        
        print(f"\nOutput files saved to: {os.path.abspath(output_dir)}")
    
    except Exception as e:
        logger.error(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
