#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Product Catalog Reconciliation Pipeline - Iteration 1

This script implements a data reconciliation pipeline that intelligently merges
product catalog data from multiple sources (Shopify export and SQL DB export),
detects discrepancies, applies weighting by field reliability, and produces a 
corrected, authoritative dataset.

Features:
- Multi-source data loading and field mapping
- Multi-stage matching (exact and fuzzy)
- Discrepancy detection and resolution
- Pattern recognition across iterations
- Quality metrics calculation
- Visualization of reconciliation results

Usage:
    python catalog_reconciliation_1.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz, process
import os
import sys
import json
import logging
import traceback
from datetime import datetime
import re
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("catalog_reconciliation")

class ProductCatalogReconciliation:
    """
    A class to handle the reconciliation of product catalog data from multiple sources.
    Implements an iterative approach with pattern recognition.
    """
    
    def __init__(self, shopify_path: str, sql_path: str, output_dir: str = None):
        """
        Initialize the reconciliation pipeline.
        
        Args:
            shopify_path: Path to Shopify export CSV
            sql_path: Path to SQL database export Excel file
            output_dir: Directory to store output files and visualizations
        """
        self.shopify_path = shopify_path
        self.sql_path = sql_path
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or f"reconciliation_output_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize dataframes
        self.shopify_data = None
        self.sql_data = None
        self.reconciled_data = None
        
        # Field mapping configurations
        self.shopify_to_standard = {
            'Handle': 'handle',
            'Title': 'title',
            'Description': 'description',
            'Vendor': 'vendor',
            'Product Category': 'category',
            'Status': 'status',
            'Variant SKU': 'sku',
            'SKU': 'sku',                   # Alternative SKU field name
            'Reference': 'sku',             # Alternative SKU field name
            'referencia': 'sku',            # Spanish version of Reference 
            'Reference Code': 'sku',        # Another possible SKU field name
            'Variant Barcode': 'barcode',
            'Variant Price': 'price',
            'Variant Image': 'image_url',
            'Variant Weight Unit': 'weight_unit'
        }
        
        self.sql_to_standard = {
            'Item': 'item_id',
            'Referencia': 'sku',
            'Reference': 'sku',            # English version
            'SKU': 'sku',                  # Standard SKU field
            'Item Code': 'sku',            # Another possible name
            'Product Code': 'sku',         # Another possible name
            'Código': 'sku',               # Spanish short version
            'Código barra principal': 'barcode',
            'Desc. item': 'title',
            'CATEGORIA': 'category',
            'MARCA': 'brand',
            'Estado': 'status',
            'U.M. invent.': 'unit_measure'
        }
        
        # Field reliability weights (higher = more reliable)
        self.field_weights = {
            'shopify': {
                'title': 0.7,
                'description': 0.9,
                'vendor': 0.8,
                'category': 0.6,
                'status': 0.5,
                'sku': 0.5,
                'barcode': 0.8,
                'price': 0.9,
                'image_url': 1.0,
                'weight_unit': 0.8
            },
            'sql': {
                'item_id': 0.9,
                'sku': 0.9,
                'barcode': 0.9,
                'title': 0.6,
                'category': 0.7,
                'brand': 0.8,
                'status': 0.7,
                'unit_measure': 0.9
            }
        }
        
        # Performance metrics
        self.metrics = {
            'total_shopify_products': 0,
            'total_sql_products': 0,
            'exact_sku_matches': 0,
            'exact_barcode_matches': 0,
            'fuzzy_matches': 0,
            'unmatched_shopify': 0,
            'unmatched_sql': 0,
            'discrepancies': {},
            'resolution_sources': {},
            'match_confidence': []
        }
        
        # Pattern storage for iterative improvement
        self.patterns = {
            'title_variations': {},
            'common_misspellings': {},
            'category_mappings': {}
        }
        
        logger.info(f"Initialized reconciliation pipeline. Output directory: {self.output_dir}")
    
    def load_data(self) -> None:
        """Load data from both sources and perform initial preprocessing"""
        logger.info(f"Loading Shopify data from {self.shopify_path}")
        self.shopify_data = pd.read_csv(self.shopify_path)
        self.metrics['total_shopify_products'] = len(self.shopify_data)
        logger.info(f"Loaded {self.metrics['total_shopify_products']} products from Shopify")
        
        logger.info(f"Loading SQL data from {self.sql_path}")
        self.sql_data = pd.read_excel(self.sql_path)
        self.metrics['total_sql_products'] = len(self.sql_data)
        logger.info(f"Loaded {self.metrics['total_sql_products']} products from SQL database")
        
        # Standardize field names
        self._standardize_field_names()
    
    def _standardize_field_names(self) -> None:
        """Map source-specific field names to standard names"""
        # For Shopify data
        shopify_cols = set(self.shopify_data.columns)
        standardized_shopify = {}
        for orig, std in self.shopify_to_standard.items():
            if orig in shopify_cols:
                standardized_shopify[orig] = std
        
        # Rename only columns that exist
        if standardized_shopify:
            self.shopify_data = self.shopify_data.rename(columns=standardized_shopify)
        
        # For SQL data
        sql_cols = set(self.sql_data.columns)
        standardized_sql = {}
        for orig, std in self.sql_to_standard.items():
            if orig in sql_cols:
                standardized_sql[orig] = std
        
        # Rename only columns that exist
        if standardized_sql:
            self.sql_data = self.sql_data.rename(columns=standardized_sql)
        
        logger.info("Standardized field names across data sources")
        
    def preprocess_data(self) -> None:
        """
        Preprocess the data to make matching more effective.
        - Clean and normalize product titles
        - Normalize barcodes and SKUs
        - Strip HTML from descriptions
        """
        # Process Shopify data
        if 'title' in self.shopify_data.columns:
            self.shopify_data['clean_title'] = self.shopify_data['title'].astype(str).str.lower().str.strip()
            self.shopify_data['clean_title'] = self.shopify_data['clean_title'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        
        # Process SQL data 
        if 'title' in self.sql_data.columns:
            self.sql_data['clean_title'] = self.sql_data['title'].astype(str).str.lower().str.strip()
            self.sql_data['clean_title'] = self.sql_data['clean_title'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        
        # IMPROVED SKU NORMALIZATION - Preserves uniqueness while enabling matching of formatting variations
        if 'sku' in self.shopify_data.columns:
            # Basic cleaning (lowercase, strip whitespace)
            self.shopify_data['normalized_sku'] = self.shopify_data['sku'].astype(str).str.strip().str.lower()
            
            # Remove non-alphanumeric characters
            self.shopify_data['normalized_sku'] = self.shopify_data['normalized_sku'].apply(
                lambda x: re.sub(r'[^\w]', '', x) if pd.notna(x) and x != 'nan' else np.nan
            )
            
            # Create alternate normalized version with zfill for potential matching
            self.shopify_data['alt_normalized_sku'] = self.shopify_data['normalized_sku'].apply(
                lambda x: x.zfill(6) if pd.notna(x) and x.isdigit() and len(x) < 6 else x
            )
            
        if 'sku' in self.sql_data.columns:
            # Basic cleaning (lowercase, strip whitespace)
            self.sql_data['normalized_sku'] = self.sql_data['sku'].astype(str).str.strip().str.lower()
            
            # Remove non-alphanumeric characters
            self.sql_data['normalized_sku'] = self.sql_data['normalized_sku'].apply(
                lambda x: re.sub(r'[^\w]', '', x) if pd.notna(x) and x != 'nan' else np.nan
            )
            
            # Create alternate normalized version with zfill for potential matching
            self.sql_data['alt_normalized_sku'] = self.sql_data['normalized_sku'].apply(
                lambda x: x.zfill(6) if pd.notna(x) and x.isdigit() and len(x) < 6 else x
            )
            
        # Normalize barcodes
        if 'barcode' in self.shopify_data.columns:
            self.shopify_data['barcode'] = self.shopify_data['barcode'].astype(str).str.strip()
            self.shopify_data['barcode'] = self.shopify_data['barcode'].replace(['nan', 'None', ''], np.nan)
        
        if 'barcode' in self.sql_data.columns:
            self.sql_data['barcode'] = self.sql_data['barcode'].astype(str).str.strip()
            # Filter out empty or invalid barcodes
            self.sql_data.loc[self.sql_data['barcode'].isin(['nan', '0', '']), 'barcode'] = np.nan
            
        logger.info("Preprocessed data for matching with improved SKU normalization")
        
    def _clean_text(self, text_series: pd.Series) -> pd.Series:
        """
        Clean and normalize text for better matching
        - Convert to lowercase
        - Remove special characters
        - Remove extra spaces
        """
        if text_series is None:
            return pd.Series()
        
        # Convert to lowercase
        cleaned = text_series.str.lower()
        # Remove special characters
        cleaned = cleaned.str.replace(r'[^\w\s]', ' ', regex=True)
        # Normalize whitespace
        cleaned = cleaned.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        return cleaned
    
    def _strip_html(self, html_text: str) -> str:
        """Remove HTML tags from text"""
        if not isinstance(html_text, str):
            return ""
        
        # Remove HTML tags
        clean_text = re.sub(r'<.*?>', ' ', html_text)
        # Remove extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def match_products(self) -> pd.DataFrame:
        """
        Match products between Shopify and SQL data using multiple methods:
        1. Direct matching on raw SKU fields (mimics Excel VLOOKUP)
        2. Exact SKU matching with original normalized SKUs
        3. Smart SKU matching with alternate normalized SKUs and title validation
        4. Exact barcode matching
        5. Fuzzy title matching
        
        Returns:
            DataFrame with match results
        """
        # Create copies to avoid modifying original data
        shopify = self.shopify_data.copy().reset_index(drop=True)
        sql = self.sql_data.copy().reset_index(drop=True)
        
        # Add source identifier columns
        shopify['source'] = 'shopify'
        sql['source'] = 'sql'
        
        # Initialize match tracking
        shopify['matched'] = False
        sql['matched'] = False
        shopify['match_method'] = None
        sql['match_method'] = None
        shopify['match_confidence'] = 0.0
        sql['match_confidence'] = 0.0
        shopify['match_id'] = np.nan
        sql['match_id'] = np.nan
        
        # Debug column names to identify SKU fields
        logger.info(f"Shopify columns: {shopify.columns.tolist()}")
        logger.info(f"SQL columns: {sql.columns.tolist()}")
        
        # Create lists to store matched pairs
        matches = []
        
        # Step 1: Direct matching on RAW SKU fields (mimics Excel VLOOKUP behavior)
        shopify_sku_field = None
        sql_sku_field = None
        
        # Look for various SKU field names in Shopify
        for field in ['sku', 'Variant SKU', 'SKU', 'Reference', 'referencia']:
            if field in shopify.columns:
                shopify_sku_field = field
                logger.info(f"Found Shopify SKU field: {shopify_sku_field}")
                break
        
        # Look for various SKU field names in SQL
        for field in ['sku', 'Referencia', 'Reference', 'SKU', 'Item Code', 'Product Code', 'Código']:
            if field in sql.columns:
                sql_sku_field = field
                logger.info(f"Found SQL SKU field: {sql_sku_field}")
                break
        
        if shopify_sku_field and sql_sku_field:
            logger.info(f"Performing direct raw SKU matching (Excel VLOOKUP equivalent) between {shopify_sku_field} and {sql_sku_field}")
            raw_sku_matches = self._match_by_field(shopify, sql, shopify_sku_field, sql_sku_field, exact=True)
            logger.info(f"Sample Shopify raw SKUs: {shopify[shopify_sku_field].dropna().head(5).tolist()}")
            logger.info(f"Sample SQL raw SKUs: {sql[sql_sku_field].dropna().head(5).tolist()}")
            
            for _, row in raw_sku_matches.iterrows():
                shopify_idx = row['shopify_index']
                sql_idx = row['sql_index']
                
                # Mark as matched
                shopify.loc[shopify_idx, 'matched'] = True
                sql.loc[sql_idx, 'matched'] = True
                shopify.loc[shopify_idx, 'match_method'] = 'raw_sku'
                sql.loc[sql_idx, 'match_method'] = 'raw_sku'
                shopify.loc[shopify_idx, 'match_confidence'] = 1.0
                sql.loc[sql_idx, 'match_confidence'] = 1.0
                
                # Generate a match ID
                match_id = f"match_{len(matches)}"
                shopify.loc[shopify_idx, 'match_id'] = match_id
                sql.loc[sql_idx, 'match_id'] = match_id
                
                # Add to matches list
                matches.append({
                    'match_id': match_id,
                    'shopify_index': shopify_idx,
                    'sql_index': sql_idx,
                    'match_method': 'raw_sku',
                    'confidence': 1.0
                })
            
            self.metrics['raw_sku_matches'] = len(raw_sku_matches)
            logger.info(f"Found {self.metrics['raw_sku_matches']} direct raw SKU matches")

        # Step 2: Match by original normalized SKU (preserves uniqueness)
        if 'normalized_sku' in shopify.columns and 'normalized_sku' in sql.columns:
            logger.info("Performing normalized SKU-based matching for remaining products")
            # Filter for unmatched products only
            shopify_unmatched = shopify[~shopify['matched']].copy().reset_index(drop=True)
            sql_unmatched = sql[~sql['matched']].copy().reset_index(drop=True)
            
            sku_matches = self._match_by_field(shopify_unmatched, sql_unmatched, 'normalized_sku', 'normalized_sku', exact=True)
            logger.info(f"Sample unmatched Shopify SKUs: {shopify_unmatched['normalized_sku'].dropna().head(5).tolist() if not shopify_unmatched.empty else []}")
            logger.info(f"Sample unmatched SQL SKUs: {sql_unmatched['normalized_sku'].dropna().head(5).tolist() if not sql_unmatched.empty else []}")
            
            for _, row in sku_matches.iterrows():
                shopify_idx = int(row['shopify_index'])
                sql_idx = int(row['sql_index'])
                
                # Get the original indices in the main dataframes
                original_shopify_idx = shopify_unmatched.iloc[shopify_idx].name
                original_sql_idx = sql_unmatched.iloc[sql_idx].name
                
                # Mark as matched
                shopify.loc[original_shopify_idx, 'matched'] = True
                sql.loc[original_sql_idx, 'matched'] = True
                shopify.loc[original_shopify_idx, 'match_method'] = 'exact_sku'
                sql.loc[original_sql_idx, 'match_method'] = 'exact_sku'
                shopify.loc[original_shopify_idx, 'match_confidence'] = 1.0
                sql.loc[original_sql_idx, 'match_confidence'] = 1.0
                
                # Generate a match ID
                match_id = f"match_{len(matches)}"
                shopify.loc[original_shopify_idx, 'match_id'] = match_id
                sql.loc[original_sql_idx, 'match_id'] = match_id
                
                # Add to matches list
                matches.append({
                    'match_id': match_id,
                    'shopify_index': original_shopify_idx,
                    'sql_index': original_sql_idx,
                    'match_method': 'exact_sku',
                    'confidence': 1.0
                })
            
            self.metrics['exact_sku_matches'] = len(sku_matches)
            logger.info(f"Found {self.metrics['exact_sku_matches']} additional exact SKU matches")
            
        # Step 3: Match by alternate normalized SKU for remaining products WITH TITLE VALIDATION
        # This handles formatting issues like missing leading zeros while preventing false matches
        if 'alt_normalized_sku' in shopify.columns and 'alt_normalized_sku' in sql.columns:
            logger.info("Performing secondary SKU matching with title validation")
            # Filter for unmatched products only
            shopify_unmatched = shopify[~shopify['matched']].copy().reset_index(drop=True)
            sql_unmatched = sql[~sql['matched']].copy().reset_index(drop=True)
            
            # For debugging - Output unmatched SKUs to see what's missing
            def output_sample_of_unmatched(df, field, source, sample_size=20):
                if field in df.columns and len(df) > 0:
                    unmatched = df[~df['matched']][field].dropna().head(sample_size).tolist()
                    logger.info(f"Sample of unmatched {source} {field}: {unmatched}")
            
            # Output a sample of unmatched SKUs from both sources
            if shopify_sku_field:
                output_sample_of_unmatched(shopify, shopify_sku_field, 'Shopify')
            if sql_sku_field:
                output_sample_of_unmatched(sql, sql_sku_field, 'SQL')
            output_sample_of_unmatched(shopify, 'normalized_sku', 'Shopify normalized')
            output_sample_of_unmatched(sql, 'normalized_sku', 'SQL normalized')
            
            # Match on alt_normalized_sku
            alt_sku_matches = self._match_by_field(
                shopify_unmatched, sql_unmatched, 
                'alt_normalized_sku', 'alt_normalized_sku', exact=True
            )
            
            # For each potential match, validate with title similarity to prevent false matches
            validated_matches = []
            
            for _, row in alt_sku_matches.iterrows():
                shopify_idx = int(row['shopify_index'])
                sql_idx = int(row['sql_index'])
                
                # Get the original indices in the main dataframes
                original_shopify_idx = shopify_unmatched.iloc[shopify_idx].name
                original_sql_idx = sql_unmatched.iloc[sql_idx].name
                
                # Get titles for validation
                shopify_title = shopify.loc[original_shopify_idx, 'clean_title'] if 'clean_title' in shopify.columns else ''
                sql_title = sql.loc[original_sql_idx, 'clean_title'] if 'clean_title' in sql.columns else ''
                
                # Calculate title similarity to validate the match
                title_similarity = fuzz.token_sort_ratio(shopify_title, sql_title) / 100.0 if shopify_title and sql_title else 0.0
                
                # If titles are somewhat similar (>30%), accept the match
                # Or if either the SKUs are exactly the same despite being in alt_normalized form
                original_shopify_sku = shopify.loc[original_shopify_idx, 'normalized_sku']
                original_sql_sku = sql.loc[original_sql_idx, 'normalized_sku']
                
                if title_similarity > 0.3 or original_shopify_sku == original_sql_sku:
                    # Mark as matched with confidence based on title similarity
                    confidence = 0.85 + (title_similarity * 0.15)  # Scale between 0.85-1.0 based on title
                    
                    shopify.loc[original_shopify_idx, 'matched'] = True
                    sql.loc[original_sql_idx, 'matched'] = True
                    shopify.loc[original_shopify_idx, 'match_method'] = 'alt_sku'
                    sql.loc[original_sql_idx, 'match_method'] = 'alt_sku'
                    shopify.loc[original_shopify_idx, 'match_confidence'] = confidence
                    sql.loc[original_sql_idx, 'match_confidence'] = confidence
                    
                    # Generate a match ID
                    match_id = f"match_{len(matches)}"
                    shopify.loc[original_shopify_idx, 'match_id'] = match_id
                    sql.loc[original_sql_idx, 'match_id'] = match_id
                    
                    # Add to matches list
                    validated_matches.append({
                        'match_id': match_id,
                        'shopify_index': original_shopify_idx,
                        'sql_index': original_sql_idx,
                        'match_method': 'alt_sku',
                        'confidence': confidence,
                        'title_similarity': title_similarity
                    })
            
            self.metrics['alt_sku_matches'] = len(validated_matches)
            matches.extend(validated_matches)
            logger.info(f"Found {self.metrics['alt_sku_matches']} alternate SKU matches with title validation")
        
        # Step 4: Match unmatched products by barcode
        if 'barcode' in shopify.columns and 'barcode' in sql.columns:
            logger.info("Performing barcode-based matching")
            # Filter for unmatched products only - using boolean indexing with filter to ensure correct indices
            shopify_unmatched = shopify[shopify['matched'] == False].copy().reset_index(drop=True)
            sql_unmatched = sql[sql['matched'] == False].copy().reset_index(drop=True)
            logger.info(f"Unmatched products for barcode matching: {len(shopify_unmatched)} Shopify, {len(sql_unmatched)} SQL")
            
            barcode_matches = self._match_by_field(
                shopify_unmatched, sql_unmatched, 'barcode', 'barcode', exact=True
            )
            
            for _, row in barcode_matches.iterrows():
                shopify_idx = row['shopify_index']
                sql_idx = row['sql_index']
                
                # Mark as matched
                shopify.loc[shopify_idx, 'matched'] = True
                sql.loc[sql_idx, 'matched'] = True
                shopify.loc[shopify_idx, 'match_method'] = 'exact_barcode'
                sql.loc[sql_idx, 'match_method'] = 'exact_barcode'
                shopify.loc[shopify_idx, 'match_confidence'] = 0.95
                sql.loc[sql_idx, 'match_confidence'] = 0.95
                
                # Generate a match ID
                match_id = f"match_{len(matches)}"
                shopify.loc[shopify_idx, 'match_id'] = match_id
                sql.loc[sql_idx, 'match_id'] = match_id
                
                # Add to matches list
                matches.append({
                    'match_id': match_id,
                    'shopify_index': shopify_idx,
                    'sql_index': sql_idx,
                    'match_method': 'exact_barcode',
                    'confidence': 0.95
                })
            
            self.metrics['exact_barcode_matches'] = len(barcode_matches)
            logger.info(f"Found {self.metrics['exact_barcode_matches']} exact barcode matches")
        
        # Step 5: Fuzzy match remaining products by title
        logger.info("Performing fuzzy title-based matching")
        # Filter for unmatched products only
        shopify_unmatched = shopify[~shopify['matched']]
        sql_unmatched = sql[~sql['matched']]
        
        if not shopify_unmatched.empty and not sql_unmatched.empty:
            if 'title_clean' in shopify_unmatched.columns and 'title_clean' in sql_unmatched.columns:
                fuzzy_matches = self._fuzzy_match_products(shopify_unmatched, sql_unmatched)
                
                for _, row in fuzzy_matches.iterrows():
                    shopify_idx = row['shopify_index']
                    sql_idx = row['sql_index']
                    confidence = row['score'] / 100  # Convert score to 0-1 range
        self.metrics['unmatched_sql'] = sql[~sql['matched']].shape[0]
        
        logger.info(f"Unmatched products: {self.metrics['unmatched_shopify']} from Shopify, "
                   f"{self.metrics['unmatched_sql']} from SQL")
        
        # Create a matches dataframe
        matches_df = pd.DataFrame(matches)
        
        # Store match confidence distribution
        if not matches_df.empty and 'confidence' in matches_df.columns:
            self.metrics['match_confidence'] = matches_df['confidence'].tolist()
        
        return matches_df
    
    def _match_by_field(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                        field1: str, field2: str, exact: bool = True) -> pd.DataFrame:
        """
        Match records between two dataframes based on field values.
        
        Args:
            df1: First dataframe (typically Shopify)
            df2: Second dataframe (typically SQL)
            field1: Field to match in df1
            field2: Field to match in df2
            exact: Whether to use exact matching (True) or fuzzy matching (False)
            
        Returns:
            DataFrame with matched indices
        """
        if field1 not in df1.columns or field2 not in df2.columns:
            logger.warning(f"Cannot match by fields: {field1} not in df1 or {field2} not in df2")
            return pd.DataFrame(columns=['shopify_index', 'sql_index', 'confidence'])
        
        # Identify which dataframe is Shopify and which is SQL
        if df1['source'].iloc[0] == 'shopify':
            shopify_df = df1
            sql_df = df2
            shopify_field = field1
            sql_field = field2
        else:
            shopify_df = df2
            sql_df = df1
            shopify_field = field2
            sql_field = field1
            
        logger.info(f"Matching by fields: {shopify_field} (Shopify) and {sql_field} (SQL)")
        
        # Create lists to store matching pairs
        matches_list = []
        
        # Check if this is a SKU field to apply special Excel VLOOKUP-like handling
        is_sku_field = any(f in field1.lower() for f in ['sku', 'reference']) or \
                      any(f in field2.lower() for f in ['sku', 'reference'])
                      
        if exact and is_sku_field:
            # Excel VLOOKUP-like matching for SKUs
            logger.info("Using Excel VLOOKUP-like matching for SKUs")
            
            # Log samples of raw values to help with debugging
            shopify_sample = [shopify_df[shopify_field].iloc[i] for i in range(min(20, len(shopify_df))) if pd.notna(shopify_df[shopify_field].iloc[i])]
            sql_sample = [sql_df[sql_field].iloc[i] for i in range(min(20, len(sql_df))) if pd.notna(sql_df[sql_field].iloc[i])]
            logger.info(f"Sample of unmatched Shopify sku: {shopify_sample}")
            logger.info(f"Sample of unmatched SQL sku: {sql_sample}")
            
            # Create lookup dictionary for SQL values with aggressive normalization
            sql_lookup = {}
            
            # First, extract and clean values from SQL dataframe with enhanced normalization
            for i, row in sql_df.iterrows():
                sql_value = row.get(sql_field)
                if pd.notna(sql_value):
                    # Convert to string and strip ALL whitespace - aggressive trim
                    sql_value_str = str(sql_value).strip()
                    if sql_value_str:
                        # Store original value
                        if sql_value_str not in sql_lookup:
                            sql_lookup[sql_value_str] = []
                        sql_lookup[sql_value_str].append(i)
                        
                        # Store without leading zeros for numeric values
                        if sql_value_str.isdigit():
                            no_leading_zeros = sql_value_str.lstrip('0')
                            if no_leading_zeros and no_leading_zeros != sql_value_str:
                                if no_leading_zeros not in sql_lookup:
                                    sql_lookup[no_leading_zeros] = []
                                sql_lookup[no_leading_zeros].append(i)
                                
                                # Also store with padded zeros to match possible formats
                                # Try padding to 6 digits which is common
                                padded = no_leading_zeros.zfill(6)
                                if padded not in sql_lookup:
                                    sql_lookup[padded] = []
                                sql_lookup[padded].append(i)
                        
                        # Store integer version for float strings (e.g., "123.0" -> "123")
                        if '.' in sql_value_str and sql_value_str.replace('.', '', 1).isdigit():
                            try:
                                int_val = str(int(float(sql_value_str)))
                                if int_val not in sql_lookup:
                                    sql_lookup[int_val] = []
                                sql_lookup[int_val].append(i)
                            except (ValueError, OverflowError):
                                pass  # Skip if conversion fails
            
            # Log the size of our lookup dictionary
            logger.info(f"Created SQL lookup dictionary with {len(sql_lookup)} unique values")
            
            # Also log normalized Shopify and SQL SKUs for comparison
            shopify_norm_sample = [str(x).strip().replace('.0', '') for x in shopify_sample]
            sql_norm_sample = [x.strip() for x in sql_sample]
            logger.info(f"Sample of unmatched Shopify normalized normalized_sku: {shopify_norm_sample}")
            logger.info(f"Sample of unmatched SQL normalized normalized_sku: {sql_norm_sample}")
            
            # For each Shopify value, find matches in the SQL lookup with enhanced handling for float values
            for i, row in shopify_df.iterrows():
                shopify_value = row.get(shopify_field)
                
                if pd.notna(shopify_value):
                    # First handle the common case where Shopify SKUs are stored as floats (e.g., 18612.0)
                    if isinstance(shopify_value, (float, np.float64, np.float32)):
                        # Convert to integer if it's a round number (which SKUs typically are)
                        if shopify_value == int(shopify_value):
                            shopify_value_str = str(int(shopify_value))
                        else:
                            shopify_value_str = str(shopify_value)
                    else:
                        # For non-float values, standard conversion and cleaning
                        shopify_value_str = str(shopify_value).strip()
                    
                    if shopify_value_str:
                        # 1. Try direct match first
                        if shopify_value_str in sql_lookup:
                            for sql_idx in sql_lookup[shopify_value_str]:
                                matches_list.append({
                                    'shopify_index': i,
                                    'sql_index': sql_idx,
                                    'confidence': 1.0
                                })
                            continue  # Found exact match, no need to try other variants
                        
                        # 2. Try with .0 suffix removed (common format in float representations)
                        if shopify_value_str.endswith('.0'):
                            no_decimal = shopify_value_str[:-2]
                            if no_decimal in sql_lookup:
                                for sql_idx in sql_lookup[no_decimal]:
                                    matches_list.append({
                                        'shopify_index': i,
                                        'sql_index': sql_idx,
                                        'confidence': 0.99
                                    })
                                continue  # Found match, no need for further variants
                        
                        # 3. Try padded zeros versions (SQL often has leading zeros)
                        # Common padding lengths are 6 or 7 digits
                        for pad_length in [6, 7]:
                            # Handle the case where float numbers have been converted to int
                            if shopify_value_str.isdigit():
                                padded = shopify_value_str.zfill(pad_length)
                                if padded in sql_lookup:
                                    for sql_idx in sql_lookup[padded]:
                                        matches_list.append({
                                            'shopify_index': i,
                                            'sql_index': sql_idx,
                                            'confidence': 0.98
                                        })
                                    continue  # Found match, no need for further variants
                            
                            # Handle case where we have a .0 suffix that needs to be removed AND padding
                            if shopify_value_str.endswith('.0'):
                                padded = shopify_value_str[:-2].zfill(pad_length)
                                if padded in sql_lookup:
                                    for sql_idx in sql_lookup[padded]:
                                        matches_list.append({
                                            'shopify_index': i,
                                            'sql_index': sql_idx,
                                            'confidence': 0.97
                                        })
                                    continue  # Found match, no need for further variants
                        
                        # 4. Try without leading zeros (for cases where SQL entry might not have leading zeros)
                        if shopify_value_str.isdigit():
                            no_leading_zeros = shopify_value_str.lstrip('0')
                            if no_leading_zeros and no_leading_zeros != shopify_value_str and no_leading_zeros in sql_lookup:
                                for sql_idx in sql_lookup[no_leading_zeros]:
                                    matches_list.append({
                                        'shopify_index': i,
                                        'sql_index': sql_idx,
                                        'confidence': 0.96  # Lower confidence
                                    })
                                continue  # Found match, no need for further variants
                        
                        # 5. Try integer version for other float-like strings that may not end with .0
                        if '.' in shopify_value_str and shopify_value_str.replace('.', '', 1).isdigit():
                            try:
                                int_val = str(int(float(shopify_value_str)))
                                if int_val in sql_lookup:
                                    for sql_idx in sql_lookup[int_val]:
                                        matches_list.append({
                                            'shopify_index': i,
                                            'sql_index': sql_idx,
                                            'confidence': 0.95  # Even lower confidence
                                        })
                                    continue  # Found match, no need for further processing
                            except (ValueError, OverflowError):
                                pass  # Skip if conversion fails
        
        elif exact:
            # Standard exact matching for non-SKU fields
            for i, row in shopify_df.iterrows():
                shopify_value = row.get(shopify_field)
                
                if pd.isna(shopify_value) or str(shopify_value).strip() == '':
                    continue
                    
                shopify_value_str = str(shopify_value).strip()
                
                for j, row2 in sql_df.iterrows():
                    sql_value = row2.get(sql_field)
                    
                    if pd.isna(sql_value) or str(sql_value).strip() == '':
                        continue
                        
                    sql_value_str = str(sql_value).strip()
                    
                    if shopify_value_str == sql_value_str:
                        matches_list.append({
                            'shopify_index': i,
                            'sql_index': j,
                            'confidence': 1.0
                        })
        else:
            # Fuzzy matching - only if both dataframes have title_clean field
            if 'title_clean' in shopify_df.columns and 'title_clean' in sql_df.columns:
                from fuzzywuzzy import process, fuzz
                
                # Get product titles
                shopify_titles = shopify_df['title_clean'].fillna('').tolist()
                sql_titles = sql_df['title_clean'].fillna('').tolist()
                
                # For each Shopify product, find best SQL match
                for idx, shopify_title in enumerate(shopify_titles):
                    # Skip empty titles
                    if not shopify_title.strip():
                        continue
                    
                    # Find best match in SQL titles
                    match_results = process.extractOne(
                        shopify_title,
                        sql_titles,
                        scorer=fuzz.token_sort_ratio,
                        score_cutoff=85  # Use a threshold of 85% similarity
                    )
                    
                    if match_results:
                        match_title, score, sql_idx = match_results
                        shopify_index = shopify_df.index[idx]
                        sql_index = sql_df.index[sql_idx]
                        
                        # Convert similarity score to confidence (0.85 to 1.0)
                        confidence = score / 100
                        
                        matches_list.append({
                            'shopify_index': shopify_index,
                            'sql_index': sql_index,
                            'confidence': confidence
                        })
        
        # Convert matches to DataFrame
        if matches_list:
            matches_df = pd.DataFrame(matches_list)
            # Log sample matches if we found some
            sample_size = min(5, len(matches_df))
            logger.info(f"Found {len(matches_df)} matches by {field1}/{field2}, sample: {matches_df.head(sample_size).to_dict('records')}")
        else:
            matches_df = pd.DataFrame(columns=['shopify_index', 'sql_index', 'confidence'])
            logger.warning(f"No matches found by {field1}/{field2}")
        
        return matches_df
    
    def detect_discrepancies(self, matches_df: pd.DataFrame) -> dict:
        """
        Detect discrepancies between matched products with likeness scores.
        
        For each matched product pair, compares fields and identifies discrepancies using
        fuzzy matching and tolerance thresholds. This enables detection of minor differences
        in text fields while allowing some tolerance for variations.
        
        Args:
            matches_df: DataFrame with matched product pairs
            
        Returns:
            Dictionary with discrepancy counts by field
        """
        from fuzzywuzzy import fuzz
        
        logger.info("Detecting discrepancies between matched products using likeness scores")
        
        # Define comparison tolerances for different field types (percentage match required)
        tolerance_thresholds = {
            'title': 85,  # Allow slight variations in titles
            'description': 75,  # More tolerance for descriptions
            'brand': 90,  # Brand names should be nearly identical
            'vendor': 90,  # Vendor names should be nearly identical
            'marca': 90,  # Spanish equivalent of brand
            'category': 80,  # Categories might have slight variations
            'categoria': 80,  # Spanish equivalent
            'weight': 100,  # Weights should match exactly
            'peso': 100,  # Spanish equivalent
            'price': 100,  # Prices should match exactly
            'barcode': 100,  # Barcodes must match exactly
            'sku': 100,  # SKUs must match exactly
            'default': 90  # Default threshold for other fields
        }
        
        # Define additional field mappings for comparison
        extended_field_mappings = {
            'brand': [('vendor', 'brand'), ('vendor', 'marca')],
            'weight': [('weight', 'weight'), ('weight', 'peso')],
            'category': [('product_type', 'category'), ('product_type', 'categoria'), ('product_type', 'linea')],
            'tags': [('tags', 'tags'), ('tags', 'etiquetas'), ('tags', 'mundo')]
        }
        
        # Initialize discrepancy tracking
        discrepancies = {}
        discrepancy_details = {}
        field_likeness_scores = {}
        
        # Initialize comparison report
        comparison_report = {
            'field_statistics': {},
            'by_field': {},
            'match_details': [],
            'summary': {}
        }
        
        # Initialize field statistics
        for field in set(self._get_standard_field_mappings().keys()).union(extended_field_mappings.keys()):
            comparison_report['field_statistics'][field] = {
                'match_count': 0,
                'mismatch_count': 0,
                'total_count': 0
            }
        
        # Process each match
        for _, match in matches_df.iterrows():
            shopify_idx = match['shopify_index']
            sql_idx = match['sql_index']
            match_id = match.get('match_id', f"match_{_}")
            
            shopify_product = self.shopify_data.loc[shopify_idx]
            sql_product = self.sql_data.loc[sql_idx]
            
            # Compare standard fields from field mapping
            standard_fields = self._get_standard_field_mappings()
            for field, (shopify_field, sql_field) in standard_fields.items():
                # Skip if either field doesn't exist
                if shopify_field not in shopify_product.index or sql_field not in sql_product.index:
                    continue
                
                # Get values and convert to string
                shopify_value = str(shopify_product[shopify_field]).strip() if not pd.isna(shopify_product[shopify_field]) else ''
                sql_value = str(sql_product[sql_field]).strip() if not pd.isna(sql_product[sql_field]) else ''
                
                # Skip if both empty
                if not shopify_value or not sql_value:
                    continue
                
                # Get threshold for this field
                threshold = tolerance_thresholds.get(field, tolerance_thresholds['default'])
                
                # Calculate likeness score using fuzzy matching
                if field in ['sku', 'barcode', 'weight', 'price']:
                    # Exact match required for critical fields
                    likeness_score = 100 if shopify_value == sql_value else 0
                else:
                    # Fuzzy match for text fields
                    likeness_score = fuzz.token_sort_ratio(shopify_value.lower(), sql_value.lower())
                
                # Track likeness score
                if field not in field_likeness_scores:
                    field_likeness_scores[field] = []
                field_likeness_scores[field].append(likeness_score)
                
                # Check for discrepancy based on threshold
                if likeness_score < threshold:
                    # Record discrepancy
                    if field not in discrepancies:
                        discrepancies[field] = 0
                        discrepancy_details[field] = []
                    
                    discrepancies[field] += 1
                    discrepancy_details[field].append({
                        'match_id': match_id,
                        'shopify_idx': int(shopify_idx),
                        'sql_idx': int(sql_idx),
                    })
                    
                    # If values are considered matching
                if likeness_score >= 80:  # 80% or higher is considered a match
                    # Initialize the field in by_field if not exists
                    if field not in comparison_report['by_field']:
                        comparison_report['by_field'][field] = {'match_count': 0, 'mismatch_count': 0, 'total_count': 0}
                    
                    # Update the match count
                    comparison_report['field_statistics'][field]['match_count'] += 1
                    comparison_report['by_field'][field]['match_count'] += 1
                else:
                    # Initialize the field in by_field if not exists
                    if field not in comparison_report['by_field']:
                        comparison_report['by_field'][field] = {'match_count': 0, 'mismatch_count': 0, 'total_count': 0}
                    
                    # Update mismatch count
                    comparison_report['field_statistics'][field]['mismatch_count'] += 1
                    comparison_report['by_field'][field]['mismatch_count'] += 1
                
                # Update total count for both dictionaries
                comparison_report['field_statistics'][field]['total_count'] += 1
                comparison_report['by_field'][field]['total_count'] += 1
                
                # For title/description, use a lower threshold but more detailed analysis
                if field in ['title', 'description'] and len(shopify_value) > 5 and len(sql_value) > 5:
                    # Extract keywords and calculate keyword overlap
                    shopify_keywords = set(re.findall(r'\b\w+\b', shopify_value.lower()))
                    sql_keywords = set(re.findall(r'\b\w+\b', sql_value.lower()))
                    
                    # Remove common stop words
                    stop_words = {'the', 'a', 'an', 'and', 'in', 'on', 'at', 'for', 'with', 'by'}
                    shopify_keywords = shopify_keywords - stop_words
                    sql_keywords = sql_keywords - stop_words
                    
                    # Calculate keyword overlap
                    if shopify_keywords and sql_keywords:  # Ensure sets are not empty
                        common_keywords = shopify_keywords.intersection(sql_keywords)
                        keyword_overlap = len(common_keywords) / max(len(shopify_keywords), len(sql_keywords)) * 100
                        
                        # Add detailed keyword analysis to field comparison
                        match_record['field_comparisons'][-1]['keyword_analysis'] = {
                            'shopify_keywords': list(shopify_keywords),
                            'sql_keywords': list(sql_keywords),
                            'common_keywords': list(common_keywords),
                            'keyword_overlap_percentage': round(keyword_overlap, 1)
                        }
                        
                        # Remove common stop words
                        stop_words = {'the', 'a', 'an', 'and', 'in', 'on', 'at', 'for', 'with', 'by'}
                        shopify_keywords = shopify_keywords - stop_words
                        sql_keywords = sql_keywords - stop_words
                        
                        # Calculate keyword overlap
                        if shopify_keywords and sql_keywords:  # Ensure sets are not empty
                            common_keywords = shopify_keywords.intersection(sql_keywords)
                            keyword_overlap = len(common_keywords) / max(len(shopify_keywords), len(sql_keywords)) * 100
                            
                            # Add detailed keyword analysis to field comparison
                            match_record['field_comparisons'][-1]['keyword_analysis'] = {
                                'shopify_keywords': list(shopify_keywords),
                                'sql_keywords': list(sql_keywords),
                                'common_keywords': list(common_keywords),
                                'keyword_overlap_percentage': round(keyword_overlap, 1)
                            }
                    
                    # If likeness score below threshold, record discrepancy
                    if likeness_score < threshold:
                        discrepancies['by_field'][field] += 1
                        discrepancies['total_count'] += 1
                        
                        # Count discrepancies by match type
                        if 'sku' in match_type.lower():
                            discrepancies['by_match_type']['sku'] += 1
                        elif 'barcode' in match_type.lower():
                            discrepancies['by_match_type']['barcode'] += 1
                        elif 'title' in match_type.lower():
                            discrepancies['by_match_type']['title'] += 1
                        
                        product_discrepancies.append({
                            'field': field,
                            'shopify_value': shopify_value_str,
                            'sql_value': sql_value_str,
                            'likeness_score': likeness_score
                        })
            
            # Add all discrepancies for this product pair to the details
            if product_discrepancies:
                discrepancies['details'].append({
                    'match_id': i,
                    'shopify_id': shopify_product.get('id', ''),
                    'sql_id': sql_product.get('id', ''),
                    'shopify_title': shopify_product.get('title', ''),
                    'sql_title': sql_product.get('title', ''),
                    'match_type': match_type,
                    'confidence': match_confidence,
                    'discrepancies': product_discrepancies
                })
            
            # Add match record to comparison report
            comparison_report['match_details'].append(match_record)
        
        # Calculate average likeness scores for each field
        avg_likeness = {}
        for field, scores in discrepancies['field_likeness_scores'].items():
            if scores:  # If we have scores for this field
                avg_likeness[field] = round(sum(scores) / len(scores), 1)
            else:
                avg_likeness[field] = None
        
        # Find fields with lowest likeness scores
        sorted_fields = sorted(
            [(field, score) for field, score in avg_likeness.items() if score is not None],
            key=lambda x: x[1]
        )
        lowest_likeness = [(field, f"{score}%") for field, score in sorted_fields[:3]]
        
        # Calculate field match percentages for comparison report
        for field in fields_to_compare:
            stats = comparison_report['field_statistics'][field]
            if stats['total_compared'] > 0:
                stats['match_percentage'] = round(stats['match_count'] / stats['total_compared'] * 100, 1)
            else:
                stats['match_percentage'] = None
        
        # Add summary statistics to comparison report
        comparison_report['summary'] = {
            'total_matches': len(matches_df),
            'matches_by_type': {
                'sku': sku_matches,
                'barcode': barcode_matches,
                'title': title_matches,
                'other': len(matches_df) - sku_matches - barcode_matches - title_matches
            },
            'discrepancies': {
                'total': discrepancies['total_count'],
                'by_field': discrepancies['by_field'],
                'by_match_type': discrepancies['by_match_type']
            },
            'average_likeness': avg_likeness,
            'lowest_likeness_fields': lowest_likeness
        }
        
        logger.info(f"Found {discrepancies['total_count']} discrepancies across {len(fields_to_compare)} fields")
        logger.info(f"Fields with lowest likeness scores: {', '.join([f'{field} ({score})' for field, score in lowest_likeness])}")
        
        # Save discrepancy details to a JSON file
        with open(os.path.join(self.output_dir, "discrepancy_details.json"), 'w') as f:
            json.dump(discrepancies, f, indent=2)
            
        # Save comprehensive comparison report to a separate file
        with open(os.path.join(self.output_dir, "field_comparison_report.json"), 'w') as f:
            json.dump(comparison_report, f, indent=2)
            
        logger.info(f"Discrepancy details saved to {os.path.join(self.output_dir, 'discrepancy_details.json')}")
        logger.info(f"Comprehensive field comparison report saved to {os.path.join(self.output_dir, 'field_comparison_report.json')}")
        
        return discrepancies
        
    def resolve_discrepancies(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve discrepancies between matched products based on field weights.
{{ ... }}
        
        Args:
            matches_df: DataFrame with matched product indices
            
        Returns:
            DataFrame with reconciled products
        """
        logger.info("Resolving discrepancies in matched products")
        
        # Fields to reconcile and their corresponding field names in each source
        fields_to_reconcile = {
            'title': ('title', 'title'),
            'description': ('description_plain', 'title'),  # Use SQL title as fallback for description
            'barcode': ('barcode', 'barcode'),
            'sku': ('sku', 'sku'),
            'category': ('category', 'category'),
            'brand': ('vendor', 'brand'),
            'status': ('status', 'status'),
            'price': ('price', None),  # Shopify only
            'image_url': ('image_url', None)  # Shopify only
        }
        
        # Create a new DataFrame to store reconciled products
        reconciled_products = []
        resolution_sources = {field: {'shopify': 0, 'sql': 0, 'combined': 0} for field in fields_to_reconcile}
        
        # Process each match
        for _, match in matches_df.iterrows():
            shopify_idx = match['shopify_index']
            sql_idx = match['sql_index']
            match_id = match['match_id']
            
            shopify_product = self.shopify_data.loc[shopify_idx].to_dict()
            sql_product = self.sql_data.loc[sql_idx].to_dict()
            
            # Start with a new product dictionary
            reconciled_product = {
                'match_id': match_id,
                'shopify_index': shopify_idx,
                'sql_index': sql_idx,
                'match_method': match['match_method'],
                'match_confidence': match['confidence']
            }
            
            # Reconcile each field
            for field, (shopify_field, sql_field) in fields_to_reconcile.items():
                # Skip if one source doesn't have this field
                if shopify_field is None or shopify_field not in shopify_product:
                    if sql_field and sql_field in sql_product and not pd.isna(sql_product[sql_field]):
                        reconciled_product[field] = sql_product[sql_field]
                        resolution_sources[field]['sql'] += 1
                    continue
                elif sql_field is None or sql_field not in sql_product:
                    if shopify_field and shopify_field in shopify_product and not pd.isna(shopify_product[shopify_field]):
                        reconciled_product[field] = shopify_product[shopify_field]
                        resolution_sources[field]['shopify'] += 1
                    continue
                
                # Get values from both sources
                shopify_value = shopify_product.get(shopify_field)
                sql_value = sql_product.get(sql_field)
                
                # If both values are missing or empty, skip
                if (pd.isna(shopify_value) or str(shopify_value).strip() == '') and \
                   (pd.isna(sql_value) or str(sql_value).strip() == ''):
                    continue
                
                # If one value is missing, use the other
                if pd.isna(shopify_value) or str(shopify_value).strip() == '':
                    reconciled_product[field] = sql_value
                    resolution_sources[field]['sql'] += 1
                    continue
                elif pd.isna(sql_value) or str(sql_value).strip() == '':
                    reconciled_product[field] = shopify_value
                    resolution_sources[field]['shopify'] += 1
                    continue
                
                # Both values exist - compare weights and resolve
                shopify_weight = self.field_weights['shopify'].get(shopify_field, 0.5)
                sql_weight = self.field_weights['sql'].get(sql_field, 0.5)
                
                # Check if values are identical
                if str(shopify_value).strip() == str(sql_value).strip():
                    reconciled_product[field] = shopify_value  # Both are the same, use either
                    resolution_sources[field]['combined'] += 1
                    continue
                
                # Use source with higher weight
                if shopify_weight > sql_weight:
                    reconciled_product[field] = shopify_value
                    resolution_sources[field]['shopify'] += 1
                elif sql_weight > shopify_weight:
                    reconciled_product[field] = sql_value
                    resolution_sources[field]['sql'] += 1
                else:
                    # Equal weights - default to Shopify as it's customer-facing
                    reconciled_product[field] = shopify_value
                    resolution_sources[field]['shopify'] += 1
            
            # Add extra fields that are unique to each source
            for field, value in shopify_product.items():
                if field not in reconciled_product and not pd.isna(value) and not field.startswith('_'):
                    reconciled_product[f'shopify_{field}'] = value
            
            for field, value in sql_product.items():
                if field not in reconciled_product and not pd.isna(value) and not field.startswith('_'):
                    reconciled_product[f'sql_{field}'] = value
            
            reconciled_products.append(reconciled_product)
        
        # Create DataFrame from reconciled products
        reconciled_df = pd.DataFrame(reconciled_products)
        
        # Store resolution sources in metrics
        self.metrics['resolution_sources'] = resolution_sources
        
        logger.info(f"Resolved discrepancies in {len(reconciled_df)} products")
        return reconciled_df
        
    def detect_patterns(self, matches_df: pd.DataFrame) -> None:
        """
        Detect patterns in the matching and discrepancies to improve future reconciliations.
        This is an iterative process that builds up pattern knowledge over time.
        
        Args:
            matches_df: DataFrame with matched product indices
        """
        logger.info("Detecting patterns in product data")
        
        # Analyze title variations
        self._detect_title_patterns(matches_df)
        
        # Analyze category mappings
        self._detect_category_mappings(matches_df)
        
        # Save pattern data for future use
        patterns_file = os.path.join(self.output_dir, 'detected_patterns.json')
        with open(patterns_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
        
        logger.info("Pattern detection complete")
        
    def _detect_title_patterns(self, matches_df: pd.DataFrame) -> None:
        """
        Detect patterns in product titles between matched products.
        
        Args:
            matches_df: DataFrame with matched product indices
        """
        title_variations = {}
        
        # Process high-confidence matches
        for _, match in matches_df.iterrows():
            if match['confidence'] < 0.9 or match['match_method'] == 'fuzzy_title':
                continue  # Skip low confidence or already fuzzy-matched titles
                
            shopify_idx = match['shopify_index']
            sql_idx = match['sql_index']
            
            # Skip if either dataframe doesn't have the necessary fields
            if 'title' not in self.shopify_data.columns or 'title' not in self.sql_data.columns:
                continue
                
            shopify_title = str(self.shopify_data.loc[shopify_idx, 'title']).strip().lower()
            sql_title = str(self.sql_data.loc[sql_idx, 'title']).strip().lower()
            
            # Skip if titles are identical or empty
            if not shopify_title or not sql_title or shopify_title == sql_title:
                continue
                
            # Calculate token variations
            shopify_tokens = set(re.findall(r'\w+', shopify_title))
            sql_tokens = set(re.findall(r'\w+', sql_title))
            
            # Find differences
            shopify_only = shopify_tokens - sql_tokens
            sql_only = sql_tokens - shopify_tokens
            
            # Record variations if they exist
            if shopify_only and sql_only:
                for s_token in shopify_only:
                    for q_token in sql_only:
                        if len(s_token) > 2 and len(q_token) > 2:  # Ignore short tokens
                            key = f"{s_token}:{q_token}"
                            if key not in title_variations:
                                title_variations[key] = 0
                            title_variations[key] += 1
        
        # Keep only patterns that appear multiple times
        title_patterns = {k: v for k, v in title_variations.items() if v > 1}
        
        # Update stored patterns
        self.patterns['title_variations'].update(title_patterns)
        
        logger.info(f"Detected {len(title_patterns)} title variation patterns")
        
    def _detect_category_mappings(self, matches_df: pd.DataFrame) -> None:
        """
        Detect patterns in product category mappings between matched products.
        
        Args:
            matches_df: DataFrame with matched product indices
        """
        category_mappings = {}
        
        # Check if both dataframes have category fields
        if 'category' not in self.shopify_data.columns or 'category' not in self.sql_data.columns:
            logger.warning("Cannot detect category mappings - missing category fields")
            return
        
        # Process high-confidence matches
        for _, match in matches_df.iterrows():
            if match['confidence'] < 0.9:
                continue  # Skip low confidence matches
                
            shopify_idx = match['shopify_index']
            sql_idx = match['sql_index']
            
            shopify_category = str(self.shopify_data.loc[shopify_idx, 'category']).strip()
            sql_category = str(self.sql_data.loc[sql_idx, 'category']).strip()
            
            # Skip if categories are identical or empty
            if not shopify_category or not sql_category or shopify_category == sql_category:
                continue
                
            # Record category mapping
            key = f"{shopify_category}:{sql_category}"
            if key not in category_mappings:
                category_mappings[key] = 0
            category_mappings[key] += 1
        
        # Keep only mappings that appear multiple times
        frequent_mappings = {k: v for k, v in category_mappings.items() if v > 1}
        
        # Update stored patterns
        self.patterns['category_mappings'].update(frequent_mappings)
        
        logger.info(f"Detected {len(frequent_mappings)} category mapping patterns")
    
    def reconcile_catalog(self) -> pd.DataFrame:
        """
        Perform the full reconciliation process:
        1. Match products across sources
        2. Detect discrepancies
        3. Resolve discrepancies
        4. Incorporate unmatched products
        5. Detect patterns for future iterations
        
        Returns:
            DataFrame with reconciled catalog
        """
        logger.info("Starting full catalog reconciliation process")
        
        # Step 1: Match products
        matches_df = self.match_products()
        
        # Step 2: Detect discrepancies
        self.detect_discrepancies(matches_df)
        
        # Step 3: Resolve discrepancies
        reconciled_matches = self.resolve_discrepancies(matches_df)
        
        # Step 4: Incorporate unmatched products
        final_catalog = self._incorporate_unmatched_products(reconciled_matches)
        
        # Step 5: Detect patterns for future iterations
        self.detect_patterns(matches_df)
        
        # Store final catalog
        self.reconciled_data = final_catalog
        
        logger.info(f"Reconciliation complete. Final catalog contains {len(final_catalog)} products")
        return final_catalog
    
    def _incorporate_unmatched_products(self, reconciled_matches: pd.DataFrame) -> pd.DataFrame:
        """
        Incorporate unmatched products from both sources into the final catalog.
        
        Args:
            reconciled_matches: DataFrame with reconciled matched products
            
        Returns:
            DataFrame with all products (matched and unmatched)
        """
        logger.info("Incorporating unmatched products into final catalog")
        
        # Get lists of matched indices
        matched_shopify_indices = reconciled_matches['shopify_index'].tolist()
        matched_sql_indices = reconciled_matches['sql_index'].tolist()
        
        # Get unmatched products
        unmatched_shopify = self.shopify_data[~self.shopify_data.index.isin(matched_shopify_indices)].copy()
        unmatched_sql = self.sql_data[~self.sql_data.index.isin(matched_sql_indices)].copy()
        
        # Process unmatched Shopify products
        unmatched_shopify_processed = []
        for idx, row in unmatched_shopify.iterrows():
            product = row.to_dict()
            processed_product = {
                'match_id': f"unmatched_shopify_{idx}",
                'source': 'shopify_only',
                'match_method': None,
                'match_confidence': 0.0
            }
            
            # Add Shopify fields with standardized names
            for field in product:
                if not field.startswith('_') and not pd.isna(product[field]):
                    processed_product[f"shopify_{field}"] = product[field]
                    
                    # Also add to standard fields if we have a mapping
                    for std_field, (shopify_field, _) in self._get_standard_field_mappings().items():
                        if field == shopify_field:
                            processed_product[std_field] = product[field]
            
            unmatched_shopify_processed.append(processed_product)
        
        # Process unmatched SQL products
        unmatched_sql_processed = []
        for idx, row in unmatched_sql.iterrows():
            product = row.to_dict()
            processed_product = {
                'match_id': f"unmatched_sql_{idx}",
                'source': 'sql_only',
                'match_method': None,
                'match_confidence': 0.0
            }
            
            # Add SQL fields with standardized names
            for field in product:
                if not field.startswith('_') and not pd.isna(product[field]):
                    processed_product[f"sql_{field}"] = product[field]
                    
                    # Also add to standard fields if we have a mapping
                    for std_field, (_, sql_field) in self._get_standard_field_mappings().items():
                        if field == sql_field:
                            processed_product[std_field] = product[field]
            
            unmatched_sql_processed.append(processed_product)
        
        # Combine all products
        all_products = (
            reconciled_matches.to_dict('records') +
            unmatched_shopify_processed +
            unmatched_sql_processed
        )
        
        # Create final DataFrame
        final_catalog = pd.DataFrame(all_products)
        
        # Add source type
        final_catalog.loc[final_catalog['match_id'].str.contains('match_'), 'source'] = 'matched'
        
        logger.info(f"Added {len(unmatched_shopify_processed)} unmatched Shopify products and "  
                   f"{len(unmatched_sql_processed)} unmatched SQL products to final catalog")
        
        return final_catalog
    
    def _get_standard_field_mappings(self) -> Dict[str, Tuple[str, str]]:
        """
        Get standardized field mappings for both sources.
        
        Returns:
            Dictionary mapping standard field names to source-specific field names
        """
        return {
            'title': ('title', 'title'),
            'description': ('description_plain', 'title'),
            'barcode': ('barcode', 'barcode'),
            'sku': ('sku', 'sku'),
            'category': ('category', 'category'),
            'brand': ('vendor', 'brand'),
        }
    
    def _plot_match_status(self, viz_dir: str) -> None:
        """
        Create a pie chart showing the distribution of match statuses.
        
        Args:
            viz_dir: Directory to save the visualization
        """
        if self.reconciled_data is None or 'source' not in self.reconciled_data.columns:
            return
            
        # Count products by source
        match_counts = self.reconciled_data['source'].value_counts()
        
        # Create pie chart
        plt.figure(figsize=(10, 6))
        plt.pie(match_counts, labels=match_counts.index, autopct='%1.1f%%', startangle=90, 
                colors=sns.color_palette('viridis', len(match_counts)))
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Product Match Status Distribution')
        
        # Save figure
        output_file = os.path.join(viz_dir, 'match_status_distribution.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
    
    def _plot_field_conflicts(self, viz_dir: str) -> None:
        """
        Create a bar chart showing fields with the most conflicts.
        
        Args:
            viz_dir: Directory to save the visualization
        """
        if 'discrepancies' not in self.metrics:
            return
            
        # Get discrepancy counts by field
        fields = list(self.metrics['discrepancies'].keys())
        counts = list(self.metrics['discrepancies'].values())
        
        # Skip if no discrepancies
        if not fields or not counts or sum(counts) == 0:
            return
            
        # Sort by count
        fields, counts = zip(*sorted(zip(fields, counts), key=lambda x: x[1], reverse=True))
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(fields, counts, color=sns.color_palette('viridis', len(fields)))
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(height)}',
                    ha='center', va='bottom', rotation=0)
        
        plt.title('Fields with Most Discrepancies')
        plt.xlabel('Field')
        plt.ylabel('Number of Discrepancies')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(viz_dir, 'field_discrepancies.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
    
    def _plot_resolution_sources(self, viz_dir: str) -> None:
        """
        Create a stacked bar chart showing resolution sources by field.
        
        Args:
            viz_dir: Directory to save the visualization
        """
        if 'resolution_sources' not in self.metrics:
            return
            
        resolution_data = self.metrics['resolution_sources']
        
        # Skip if no resolution data
        if not resolution_data:
            return
            
        # Prepare data for plotting
        fields = list(resolution_data.keys())
        shopify_counts = [resolution_data[field]['shopify'] for field in fields]
        sql_counts = [resolution_data[field]['sql'] for field in fields]
        combined_counts = [resolution_data[field]['combined'] for field in fields]
        
        # Create stacked bar chart
        plt.figure(figsize=(12, 6))
        
        # Create bars
        bar_width = 0.6
        plt.bar(fields, shopify_counts, bar_width, label='Shopify', color='#5B9BD5')
        plt.bar(fields, sql_counts, bar_width, bottom=shopify_counts, label='SQL', color='#ED7D31')
        plt.bar(fields, combined_counts, bar_width, 
                bottom=[i+j for i,j in zip(shopify_counts, sql_counts)], 
                label='Combined (identical)', color='#70AD47')
        
        plt.title('Resolution Source by Field')
        plt.xlabel('Field')
        plt.ylabel('Count')
        plt.legend(loc='upper right')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(viz_dir, 'resolution_sources.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
    
    def _plot_category_distribution(self, viz_dir: str) -> None:
        """
        Create a bar chart showing the distribution of product categories.
        
        Args:
            viz_dir: Directory to save the visualization
        """
        if self.reconciled_data is None or 'category' not in self.reconciled_data.columns:
            return
            
        # Count products by category
        category_counts = self.reconciled_data['category'].value_counts().head(15)  # Top 15 categories
        
        # Skip if no categories
        if len(category_counts) == 0:
            return
            
        # Create horizontal bar chart for better label readability
        plt.figure(figsize=(12, 8))
        bars = plt.barh(category_counts.index, category_counts.values, 
                      color=sns.color_palette('viridis', len(category_counts)))
        
        # Add value labels at end of bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                    ha='left', va='center')
        
        plt.title('Top Categories in Reconciled Catalog')
        plt.xlabel('Number of Products')
        plt.ylabel('Category')
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(viz_dir, 'category_distribution.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
    def evaluate_reconciliation(self) -> Dict[str, Any]:
        """
        Evaluate the quality of the reconciliation process.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.reconciled_data is None:
            logger.error("No reconciled data available. Run reconcile_catalog() first.")
            return {}
            
        logger.info("Evaluating reconciliation quality")
        
        # Calculate evaluation metrics
        total_products = len(self.reconciled_data)
        matched_products = self.metrics['exact_sku_matches'] + self.metrics['exact_barcode_matches'] + self.metrics['fuzzy_matches']
        match_rate = matched_products / (self.metrics['total_shopify_products'] + self.metrics['total_sql_products'] - matched_products) if matched_products > 0 else 0
        
        # Calculate average match confidence
        match_confidence = np.mean(self.metrics['match_confidence']) if self.metrics['match_confidence'] else 0
        
        # Calculate resolution breakdown
        if 'resolution_sources' in self.metrics:
            resolution_counts = {
                'shopify': sum(source['shopify'] for source in self.metrics['resolution_sources'].values()),
                'sql': sum(source['sql'] for source in self.metrics['resolution_sources'].values()),
                'combined': sum(source['combined'] for source in self.metrics['resolution_sources'].values())
            }
            total_resolutions = sum(resolution_counts.values())
            resolution_percentages = {
                k: v / total_resolutions * 100 if total_resolutions > 0 else 0 
                for k, v in resolution_counts.items()
            }
        else:
            resolution_percentages = {'shopify': 0, 'sql': 0, 'combined': 0}
        
        # Compile evaluation results
        evaluation = {
            'total_products': total_products,
            'matched_products': matched_products,
            'match_rate': match_rate,
            'match_confidence': match_confidence,
            'resolution_percentages': resolution_percentages,
            'unmatched_shopify': self.metrics['unmatched_shopify'],
            'unmatched_sql': self.metrics['unmatched_sql']
        }
        
        # Save evaluation metrics
        eval_file = os.path.join(self.output_dir, 'evaluation_metrics.json')
        with open(eval_file, 'w') as f:
            json.dump({k: float(v) if isinstance(v, np.float64) else v for k, v in evaluation.items()}, f, indent=2)
        
        logger.info(f"Saved evaluation metrics to {eval_file}")
        return evaluation


def main():
    """
    Main function to run the reconciliation pipeline.
    """
    # Configure logging to console for visibility with timestamps
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Also log to a file for debugging
    file_handler = logging.FileHandler('data_output/reconciliation_debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    print("STARTING RECONCILIATION PIPELINE")
    sys.stdout.flush()
    
    # Configure paths
    shopify_path = os.path.join('files', 'Shopify_products_export_24_07_2025.csv')
    sql_path = os.path.join('data_output', 'ITEMS_Supermu_SQL_copy.xlsx')
    output_dir = os.path.join('data_output')
    
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
    
    print(f"Shopify file exists: {os.path.exists(shopify_path)}")
    print(f"SQL file exists: {os.path.exists(sql_path)}")
    print(f"Shopify file size: {os.path.getsize(shopify_path) / (1024*1024):.2f} MB")
    print(f"SQL file size: {os.path.getsize(sql_path) / (1024*1024):.2f} MB")
    sys.stdout.flush()
    
    output_dir = "data_output"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Initialize and run pipeline
        pipeline = ProductCatalogReconciliation(shopify_path, sql_path, output_dir)
        
        print("\n1. Loading data from both sources...")
        pipeline.load_data()
        
        print("\n2. Preprocessing data...")
        pipeline.preprocess_data()
        
        print("\n3. Reconciling catalog data...")
        sys.stdout.flush()
        try:
            logger.info("Starting reconcile_catalog() method")
            reconciled_data = pipeline.reconcile_catalog()
            logger.info(f"Reconcile_catalog() completed successfully with {len(reconciled_data)} products")
        except Exception as e:
            logger.error(f"Error in reconcile_catalog: {str(e)}")
            traceback.print_exc()
            raise
            
        print("\n4. Generating output files and visualizations...")
        sys.stdout.flush()
        try:
            logger.info("Starting generate_output() method")
            pipeline.generate_output()
            logger.info("Output generation completed successfully")
        except Exception as e:
            logger.error(f"Error in generate_output: {str(e)}")
            traceback.print_exc()
            raise
        
        print("\n5. Evaluating reconciliation quality...")
        sys.stdout.flush()
        try:
            logger.info("Starting evaluate_reconciliation() method")
            evaluation = pipeline.evaluate_reconciliation()
            logger.info("Evaluation completed successfully")
        except Exception as e:
            logger.error(f"Error in evaluate_reconciliation: {str(e)}")
            traceback.print_exc()
            raise
        
        # Print summary
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
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
