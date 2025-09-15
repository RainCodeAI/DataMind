# Schema Builder v1 Specification

## Overview
Automated semantic schema generation from CSV uploads with PII detection, type inference, and relationship mapping.

## Core Components

### CSV Upload & Validation
- File size limit: 100MB
- Supported formats: CSV, TSV
- Encoding detection and normalization
- Malformed data handling and reporting

### Data Profiling Pipeline
1. **Type Detection**: Statistical analysis for string/numeric/datetime/boolean
2. **Category Classification**: identifier, metric, temporal, categorical
3. **Unit Detection**: Currency, percentages, measurements
4. **Synonym Mapping**: Common aliases and variations
5. **PII Detection**: Regex patterns + sample data analysis

### Semantic Schema Output
```json
{
  "columns": [
    {
      "name": "customer_name",
      "type": "string",
      "category": "identifier", 
      "pii_detected": true,
      "blocked": true,
      "synonyms": ["name", "client_name"],
      "confidence": 0.95
    }
  ],
  "relationships": [],
  "constraints": {
    "max_rows": 1000,
    "blocked_columns": ["customer_name"],
    "allowed_operations": ["SELECT", "WHERE", "GROUP BY", "ORDER BY"]
  }
}
```

## Performance Requirements
- Profiling time: < 30 seconds for 1M rows
- Memory usage: < 1GB peak
- Accuracy: >95% type detection, >90% PII detection

## Security Constraints
- PII deny-list enforcement
- Read-only schema access
- Audit trail for all profiling operations