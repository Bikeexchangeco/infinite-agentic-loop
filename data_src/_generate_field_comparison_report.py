def _generate_field_comparison_report(self) -> Dict[str, Any]:
    """
    Generate a detailed field comparison report for matched products.
    This report helps the LLM analyze field discrepancies and matching quality.
    
    Returns:
        Dictionary with field comparison statistics and details
    """
    logger.info("Generating field comparison report")
    
    # Initialize field comparison report
    report = {
        "summary": {
            "total_matches": 0,
            "matches_by_type": {},
            "field_match_rates": {}
        },
        "field_statistics": {},
        "match_details": []
    }
    
    # Get matched products
    matched_products = self.reconciled_data[self.reconciled_data['source'] == 'both'].copy()
    report["summary"]["total_matches"] = len(matched_products)
    
    if len(matched_products) == 0:
        logger.warning("No matched products found for field comparison report")
        return report
        
    # Count matches by method/type
    if 'match_method' in matched_products.columns:
        method_counts = matched_products['match_method'].value_counts().to_dict()
        report["summary"]["matches_by_type"] = method_counts
    
    # Fields to analyze (get all fields in the reconciled catalog)
    all_fields = []
    shopify_fields = [col for col in matched_products.columns if col.startswith('shopify_')]
    sql_fields = [col for col in matched_products.columns if col.startswith('sql_')]
    
    # Map Shopify fields to SQL fields (based on common suffix after the source prefix)
    field_pairs = {}
    for sf in shopify_fields:
        sf_name = sf.replace('shopify_', '')
        for sqlf in sql_fields:
            sqlf_name = sqlf.replace('sql_', '')
            if sf_name == sqlf_name:
                field_pairs[sf_name] = (sf, sqlf)
                all_fields.append(sf_name)
                break
    
    # Analyze each field pair
    for field_name, (shopify_field, sql_field) in field_pairs.items():
        # Calculate match statistics
        field_stats = {
            "field_name": field_name,
            "shopify_field": shopify_field,
            "sql_field": sql_field,
            "match_count": 0,
            "mismatch_count": 0,
            "match_rate": 0.0,
            "avg_similarity": 0.0,
            "null_count": 0
        }
        
        # Extract values for comparison
        shopify_values = matched_products[shopify_field].fillna('')
        sql_values = matched_products[sql_field].fillna('')
        
        # Compare values
        matches = 0
        mismatches = 0
        similarities = []
        nulls = 0
        
        for i, (s_val, sql_val) in enumerate(zip(shopify_values, sql_values)):
            # Handle nulls
            if pd.isna(s_val) and pd.isna(sql_val):
                nulls += 1
                continue
            
            # Convert to string for comparison
            s_val_str = str(s_val) if not pd.isna(s_val) else ""
            sql_val_str = str(sql_val) if not pd.isna(sql_val) else ""
            
            # Calculate similarity based on field type
            if field_name in ['title', 'description', 'category', 'tags']:
                # For text fields, use token sort ratio
                similarity = fuzz.token_sort_ratio(s_val_str, sql_val_str) / 100.0
            else:
                # For other fields, use simple equality check
                similarity = 1.0 if s_val_str.strip() == sql_val_str.strip() else 0.0
            
            similarities.append(similarity)
            
            if similarity > 0.9:  # Threshold for considering a match
                matches += 1
            else:
                mismatches += 1
        
        # Calculate statistics
        total = matches + mismatches
        if total > 0:
            field_stats["match_count"] = matches
            field_stats["mismatch_count"] = mismatches
            field_stats["match_rate"] = matches / total
            field_stats["avg_similarity"] = sum(similarities) / len(similarities) if similarities else 0.0
            field_stats["null_count"] = nulls
            
            # Add to summary
            report["summary"]["field_match_rates"][field_name] = field_stats["match_rate"]
        
        # Add to field statistics
        report["field_statistics"][field_name] = field_stats
    
    # Generate detailed match information for sample of matched products
    sample_size = min(50, len(matched_products))
    sample_indices = matched_products.index[:sample_size]
    
    for idx in sample_indices:
        product = matched_products.loc[idx]
        
        match_detail = {
            "match_id": product.get('match_id', f"match_{idx}"),
            "match_type": product.get('match_method', 'unknown'),
            "confidence": float(product.get('match_confidence', 0.0)),
            "fields": {}
        }
        
        # Add field comparisons
        for field_name, (shopify_field, sql_field) in field_pairs.items():
            # Skip internal fields
            if field_name.startswith('_') or field_name in ['matched', 'source']:
                continue
                
            shopify_value = product.get(shopify_field, '')
            sql_value = product.get(sql_field, '')
            
            # Convert to string for comparison
            shopify_str = str(shopify_value) if not pd.isna(shopify_value) else ""
            sql_str = str(sql_value) if not pd.isna(sql_value) else ""
            
            # Calculate similarity
            if field_name in ['title', 'description', 'category', 'tags']:
                # For text fields, use token sort ratio
                similarity = fuzz.token_sort_ratio(shopify_str, sql_str) / 100.0
                
                # For title and description, also calculate keyword overlap
                if field_name in ['title', 'description'] and shopify_str and sql_str:
                    # Extract keywords (simple tokenization)
                    shopify_keywords = set(re.findall(r'\b\w+\b', shopify_str.lower()))
                    sql_keywords = set(re.findall(r'\b\w+\b', sql_str.lower()))
                    
                    # Calculate overlap
                    if shopify_keywords and sql_keywords:
                        overlap = len(shopify_keywords.intersection(sql_keywords))
                        keyword_overlap_pct = overlap / max(len(shopify_keywords), len(sql_keywords))
                    else:
                        keyword_overlap_pct = 0.0
                else:
                    keyword_overlap_pct = None
            else:
                # For other fields, use simple equality check
                similarity = 1.0 if shopify_str.strip() == sql_str.strip() else 0.0
                keyword_overlap_pct = None
            
            field_detail = {
                "shopify_value": shopify_str,
                "sql_value": sql_str,
                "similarity": similarity,
                "is_match": similarity > 0.9  # Threshold for considering a match
            }
            
            # Add keyword overlap if applicable
            if keyword_overlap_pct is not None:
                field_detail["keyword_overlap_pct"] = keyword_overlap_pct
            
            match_detail["fields"][field_name] = field_detail
        
        report["match_details"].append(match_detail)
    
    return report
