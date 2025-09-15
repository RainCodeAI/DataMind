# NL→SQL Engine Specification

## Overview
Constrained natural language to SQL conversion with security guards and validation.

## Input Processing
1. **Interceptor**: Prompt sanitization and intent classification
2. **Context Building**: Dataset schema + user context
3. **LLM Generation**: Structured prompt → SQL generation
4. **Validation**: sqlglot parsing and security checks

## Security Constraints
- **SELECT-only**: No DDL/DML operations
- **LIMIT 1000**: Mandatory row limit enforcement
- **Column Deny-lists**: PII and sensitive data blocking
- **Complexity Limits**: Query complexity scoring
- **Schema Validation**: Table/column access verification

## Output Validation (Woodpecker)
1. **AST Analysis**: Parse and validate SQL structure
2. **Schema Consistency**: Verify column/table references
3. **Aggregation Safety**: Check aggregation logic
4. **Performance Estimation**: Row count and cost analysis
5. **Result Sanity**: Basic result validation

## LLM Integration
- **Model**: GPT-4 or Claude-3.5 for reliability
- **Prompt Engineering**: Few-shot examples with schema context
- **Confidence Scoring**: 0-1 confidence with thresholds
- **Error Handling**: Graceful degradation and user feedback

## Performance Requirements
- Query generation: < 2 seconds
- Validation: < 500ms
- Overall pipeline: < 5 seconds p95
- Accuracy: ≥95% on canonical test suite

## Supported Query Types
- Basic aggregations (SUM, COUNT, AVG)
- Grouping and filtering
- Time series analysis
- Top-N queries
- Simple joins (future)