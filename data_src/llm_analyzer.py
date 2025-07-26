#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Analysis Module for Product Catalog Reconciliation
-----------------------------------------------------
This module integrates with LLMs (such as Claude) to analyze reconciliation 
output files and generate insights and recommendations for improving data quality
and matching processes.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import anthropic  # Requires installation: pip install anthropic
from typing import Dict, List, Any, Optional, Union, Tuple
import time
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_output/llm_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """LLM-powered analysis of catalog reconciliation results."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "claude-3-sonnet-20240229",
                 output_dir: str = "data_output"):
        """
        Initialize the LLM Analyzer.
        
        Args:
            api_key: Optional API key for the LLM service
            model: LLM model to use
            output_dir: Directory for output files
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Will run in 'mock' mode without actual LLM calls.")
        
        self.model = model
        self.output_dir = output_dir
        self.client = None
        
        # Try to initialize the Anthropic client if API key is available
        if self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Initialized Anthropic client with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {str(e)}")
                self.api_key = None
    
    def analyze_reconciliation_output(self) -> Dict[str, Any]:
        """
        Main entry point: Analyze all reconciliation output files and generate insights.
        
        Returns:
            Dictionary containing analysis results and recommendations
        """
        logger.info("Starting LLM analysis of reconciliation output")
        
        # Load all relevant output files
        data = self._load_output_files()
        if not data:
            logger.error("No data loaded for analysis")
            return {"error": "No data loaded for analysis"}
        
        # Generate various analyses
        results = {}
        
        # Analysis 1: SKU matching analysis
        results["sku_matching_analysis"] = self._analyze_sku_matching(data)
        
        # Analysis 2: Field discrepancy analysis
        results["field_discrepancy_analysis"] = self._analyze_field_discrepancies(data)
        
        # Analysis 3: Data quality assessment
        results["data_quality_assessment"] = self._analyze_data_quality(data)
        
        # Analysis 4: Reconciliation effectiveness
        results["reconciliation_effectiveness"] = self._analyze_reconciliation_effectiveness(data)
        
        # Analysis 5: Recommendations for improvement
        results["recommendations"] = self._generate_recommendations(data, results)
        
        # Save analysis results
        self._save_analysis_results(results)
        
        logger.info("LLM analysis completed")
        return results

    def _load_output_files(self) -> Dict[str, Any]:
        """Load all relevant reconciliation output files."""
        data = {}
        
        # List of files to load with their keys
        files_to_load = [
            ("discrepancy_details.json", "discrepancies"),
            ("reconciliation_metrics.json", "metrics"),
            ("field_comparison_report.json", "field_comparison"),
            ("evaluation_metrics.json", "evaluation")
        ]
        
        for filename, key in files_to_load:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data[key] = json.load(f)
                    logger.info(f"Loaded {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
            else:
                logger.warning(f"File not found: {filepath}")
        
        # Load a sample of the reconciled catalog as DataFrame (if exists)
        catalog_path = os.path.join(self.output_dir, "reconciled_catalog.csv")
        if os.path.exists(catalog_path):
            try:
                # Load only a sample for efficiency
                data["catalog_sample"] = pd.read_csv(catalog_path, nrows=100).to_dict(orient="records")
                logger.info("Loaded sample of reconciled catalog")
            except Exception as e:
                logger.error(f"Error loading reconciled_catalog.csv: {str(e)}")
        
        return data
    
    def _analyze_sku_matching(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze SKU matching patterns and issues.
        
        Args:
            data: Dictionary containing loaded reconciliation data
            
        Returns:
            Dictionary with SKU matching analysis results
        """
        logger.info("Analyzing SKU matching patterns")
        
        # Extract relevant SKU matching data
        sku_data = {}
        if "field_comparison" in data:
            field_comp = data["field_comparison"]
            if "summary" in field_comp and "matches_by_type" in field_comp["summary"]:
                sku_data["match_counts"] = field_comp["summary"]["matches_by_type"]
            
            # Extract sample matches for analysis
            if "match_details" in field_comp:
                sku_matches = [m for m in field_comp["match_details"] 
                              if m.get("match_type", "").lower().startswith("sku")]
                sku_data["sample_matches"] = sku_matches[:10]  # First 10 matches
        
        # If we have discrepancies data, extract SKU-related issues
        if "discrepancies" in data:
            disc = data["discrepancies"]
            sku_issues = [d for d in disc.get("details", []) 
                         if any(disc.get("field") == "sku" for disc in d.get("discrepancies", []))]
            sku_data["sku_issues"] = sku_issues[:10]  # First 10 issues
        
        # Now use LLM to analyze this data
        analysis = self._llm_analyze(
            data_subset=sku_data,
            analysis_type="sku_matching",
            prompt="""
            Analyze the SKU matching patterns in this reconciliation data. Focus on:
            1. Why are some SKUs matching while others aren't?
            2. What patterns do you see in the SKUs that successfully match?
            3. What formatting inconsistencies exist between the data sources?
            4. What recommendations would you make to improve SKU matching?
            
            Provide a detailed analysis with specific examples from the data.
            """
        )
        
        return {
            "raw_data": sku_data,
            "llm_analysis": analysis,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _analyze_field_discrepancies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze discrepancies between fields in matched products.
        
        Args:
            data: Dictionary containing loaded reconciliation data
            
        Returns:
            Dictionary with field discrepancy analysis results
        """
        logger.info("Analyzing field discrepancies")
        
        # Extract relevant discrepancy data
        discrepancy_data = {}
        if "discrepancies" in data:
            disc = data["discrepancies"]
            discrepancy_data["by_field"] = disc.get("by_field", {})
            discrepancy_data["sample_details"] = disc.get("details", [])[:10]
            
            # Get average likeness scores if available
            if "field_likeness_scores" in disc:
                scores = {}
                for field, values in disc["field_likeness_scores"].items():
                    if values:
                        scores[field] = sum(values) / len(values)
                discrepancy_data["avg_likeness_scores"] = scores
        
        # Use LLM to analyze this data
        analysis = self._llm_analyze(
            data_subset=discrepancy_data,
            analysis_type="field_discrepancies",
            prompt="""
            Analyze the field discrepancies between matched products. Focus on:
            1. Which fields show the highest discrepancy rates?
            2. What patterns do you see in the discrepancies?
            3. Are the discrepancies systematic or random?
            4. How might these discrepancies impact business operations?
            5. What strategies would you recommend for resolving these discrepancies?
            
            Provide specific examples and patterns from the data.
            """
        )
        
        return {
            "raw_data": discrepancy_data,
            "llm_analysis": analysis,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _analyze_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze overall data quality in both sources.
        
        Args:
            data: Dictionary containing loaded reconciliation data
            
        Returns:
            Dictionary with data quality assessment results
        """
        logger.info("Analyzing data quality")
        
        # Extract relevant quality metrics
        quality_data = {}
        
        # Sample from catalog if available
        if "catalog_sample" in data:
            quality_data["catalog_sample"] = data["catalog_sample"]
        
        # Field statistics if available
        if "field_comparison" in data and "field_statistics" in data["field_comparison"]:
            quality_data["field_stats"] = data["field_comparison"]["field_statistics"]
        
        # Use LLM to analyze this data
        analysis = self._llm_analyze(
            data_subset=quality_data,
            analysis_type="data_quality",
            prompt="""
            Analyze the data quality in the product catalog. Focus on:
            1. What data quality issues are evident in each source?
            2. Are there missing values, inconsistent formats, or other quality issues?
            3. Which fields have the highest/lowest quality?
            4. What patterns of data quality issues do you observe?
            5. What data governance recommendations would you make?
            
            Provide specific examples and quality metrics from the data.
            """
        )
        
        return {
            "raw_data": quality_data,
            "llm_analysis": analysis,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _analyze_reconciliation_effectiveness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the effectiveness of the reconciliation process.
        
        Args:
            data: Dictionary containing loaded reconciliation data
            
        Returns:
            Dictionary with reconciliation effectiveness analysis
        """
        logger.info("Analyzing reconciliation effectiveness")
        
        # Extract relevant metrics
        effectiveness_data = {}
        
        # Overall metrics
        if "metrics" in data:
            effectiveness_data["overall_metrics"] = data["metrics"]
        
        # Evaluation metrics if available
        if "evaluation" in data:
            effectiveness_data["evaluation"] = data["evaluation"]
        
        # Match summary if available
        if "field_comparison" in data and "summary" in data["field_comparison"]:
            effectiveness_data["match_summary"] = data["field_comparison"]["summary"]
        
        # Use LLM to analyze this data
        analysis = self._llm_analyze(
            data_subset=effectiveness_data,
            analysis_type="reconciliation_effectiveness",
            prompt="""
            Analyze the effectiveness of the product catalog reconciliation process. Focus on:
            1. How effective was the matching process overall?
            2. Which matching methods (SKU, barcode, title) were most effective?
            3. What percentage of products were successfully reconciled?
            4. What are the major limitations in the current reconciliation approach?
            5. How could the reconciliation process be improved?
            
            Provide specific metrics and insights from the data.
            """
        )
        
        return {
            "raw_data": effectiveness_data,
            "llm_analysis": analysis,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _generate_recommendations(self, data: Dict[str, Any], analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall recommendations based on all analyses.
        
        Args:
            data: Dictionary containing loaded reconciliation data
            analyses: Dictionary containing all previous analysis results
            
        Returns:
            Dictionary with recommendations
        """
        logger.info("Generating recommendations")
        
        # Prepare a summary of the analyses for the LLM
        analyses_summary = {}
        for key, analysis in analyses.items():
            if key != "recommendations" and "llm_analysis" in analysis:
                analyses_summary[key] = analysis["llm_analysis"]
        
        # Use LLM to generate comprehensive recommendations
        recommendations = self._llm_analyze(
            data_subset={"analyses": analyses_summary},
            analysis_type="recommendations",
            prompt="""
            Based on all the analyses performed, provide comprehensive recommendations for:
            
            1. Improving data quality in both source systems (Shopify and SQL database)
            2. Enhancing the SKU matching algorithm and addressing format inconsistencies
            3. Resolving field discrepancies systematically
            4. Implementing data governance practices to prevent future issues
            5. Monitoring and maintaining data quality over time
            6. Specific changes to implement in the reconciliation pipeline
            
            For each recommendation:
            - Provide a clear, actionable recommendation
            - Explain the rationale based on the analysis
            - Include implementation considerations
            - Describe the expected business impact
            
            Prioritize recommendations based on expected impact and implementation difficulty.
            """
        )
        
        return {
            "llm_recommendations": recommendations,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _llm_analyze(self, 
                    data_subset: Dict[str, Any], 
                    analysis_type: str, 
                    prompt: str) -> str:
        """
        Analyze data using LLM.
        
        Args:
            data_subset: Dictionary with relevant data for this analysis
            analysis_type: Type of analysis being performed
            prompt: Prompt to send to the LLM
            
        Returns:
            String containing LLM analysis response
        """
        logger.info(f"Running LLM analysis for: {analysis_type}")
        
        # If no API key or client, run in mock mode
        if not self.api_key or not self.client:
            logger.warning("Running in mock mode without LLM")
            return self._generate_mock_analysis(analysis_type)
        
        try:
            # Format the data as string (limit to reasonable size)
            data_str = json.dumps(data_subset, indent=2)
            if len(data_str) > 50000:  # Truncate if too large
                logger.warning(f"Data for {analysis_type} exceeds 50K chars; truncating")
                data_str = data_str[:50000] + "... [truncated]"
            
            # Construct the full prompt
            full_prompt = f"""
            # Catalog Reconciliation Analysis: {analysis_type}

            ## Data
            ```json
            {data_str}
            ```

            ## Analysis Request
            {prompt}

            ## Response Format
            Provide your analysis in a clear, structured format with headings, bullet points, and examples where appropriate.
            Include specific insights from the data and actionable recommendations.
            """
            
            # Make the API call to Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            # Extract the response
            analysis = response.content[0].text
            logger.info(f"Received LLM analysis for {analysis_type} ({len(analysis)} chars)")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error during LLM analysis for {analysis_type}: {str(e)}")
            return f"Analysis failed with error: {str(e)}"
    
    def _generate_mock_analysis(self, analysis_type: str) -> str:
        """Generate mock analysis when LLM is not available."""
        logger.info(f"Generating mock analysis for {analysis_type}")
        
        mock_analyses = {
            "sku_matching": """
            # SKU Matching Analysis
            
            ## Key Findings
            * Significant format inconsistencies between Shopify and SQL SKUs
            * Shopify SKUs often stored as floats (e.g., 18612.0) while SQL SKUs have padded zeros and whitespace
            * Only a small percentage of SKUs are successfully matched
            * Leading zeros handling appears to be a major issue
            
            ## Recommendations
            1. Standardize SKU formats across systems
            2. Implement more flexible matching algorithms
            3. Consider a business rule to enforce SKU format consistency
            """,
            
            "field_discrepancies": """
            # Field Discrepancy Analysis
            
            ## Key Findings
            * Highest discrepancy rates in: SKU, barcode, and category fields
            * Many discrepancies appear systematic rather than random
            * Title fields show significant variation in format and content
            * Brand/vendor information often inconsistent between sources
            
            ## Recommendations
            1. Standardize field naming and formats across systems
            2. Implement data validation rules
            3. Create field-specific normalization logic
            """,
            
            "data_quality": """
            # Data Quality Assessment
            
            ## Key Findings
            * Overall data quality is mixed with significant inconsistencies
            * Missing values in critical fields
            * Inconsistent formatting across multiple fields
            * Text fields contain varied formats and conventions
            
            ## Recommendations
            1. Implement data quality checks in source systems
            2. Create data governance policies
            3. Regular data quality audits
            """,
            
            "reconciliation_effectiveness": """
            # Reconciliation Effectiveness Analysis
            
            ## Key Findings
            * Current matching rate is significantly lower than expected
            * Barcode matching more reliable than SKU matching in current implementation
            * Many products remain unmatched or incorrectly matched
            
            ## Recommendations
            1. Enhance matching algorithms
            2. Implement multi-stage matching with confidence scoring
            3. Add machine learning components for fuzzy matching
            """,
            
            "recommendations": """
            # Comprehensive Recommendations
            
            ## High Priority
            1. **Standardize SKU Format**: Implement consistent SKU formatting rules across all systems
            2. **Enhance Matching Logic**: Add more sophisticated normalization and comparison algorithms
            3. **Data Validation**: Implement validation rules in both source systems
            
            ## Medium Priority
            1. **Field Mapping Improvements**: Create more robust field mappings with alias handling
            2. **Data Governance**: Establish data standards and governance processes
            3. **Monitoring**: Implement ongoing data quality monitoring
            
            ## Technical Implementation
            1. Refine SKU normalization to handle floats, padding, and whitespace
            2. Add confidence scoring to all matches
            3. Implement machine learning for fuzzy matching of product attributes
            """
        }
        
        return mock_analyses.get(analysis_type, "Mock analysis not available for this type")
    
    def _save_analysis_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to file."""
        output_path = os.path.join(self.output_dir, "llm_analysis_results.json")
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved analysis results to {output_path}")
            
            # Also generate a markdown report for human readability
            self._generate_markdown_report(results)
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> None:
        """Generate a human-readable markdown report from analysis results."""
        output_path = os.path.join(self.output_dir, "llm_analysis_report.md")
        
        try:
            with open(output_path, 'w') as f:
                f.write("# Catalog Reconciliation Analysis Report\n\n")
                f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Add each analysis section
                for section, data in results.items():
                    if section != "recommendations" and "llm_analysis" in data:
                        f.write(f"## {section.replace('_', ' ').title()}\n\n")
                        f.write(data["llm_analysis"])
                        f.write("\n\n---\n\n")
                
                # Add recommendations at the end
                if "recommendations" in results and "llm_recommendations" in results["recommendations"]:
                    f.write("## Recommendations\n\n")
                    f.write(results["recommendations"]["llm_recommendations"])
            
            logger.info(f"Generated markdown report at {output_path}")
        
        except Exception as e:
            logger.error(f"Error generating markdown report: {str(e)}")


def run_llm_analysis(api_key: Optional[str] = None, output_dir: str = "data_output"):
    """
    Run the LLM analysis on reconciliation output files.
    
    Args:
        api_key: Optional API key for the LLM service
        output_dir: Directory containing output files to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = LLMAnalyzer(api_key=api_key, output_dir=output_dir)
    results = analyzer.analyze_reconciliation_output()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Analysis of Catalog Reconciliation Results")
    parser.add_argument("--api-key", type=str, help="API key for the LLM service (defaults to ANTHROPIC_API_KEY env var)")
    parser.add_argument("--output-dir", type=str, default="data_output", help="Directory containing reconciliation output files")
    
    args = parser.parse_args()
    
    run_llm_analysis(api_key=args.api_key, output_dir=args.output_dir)
