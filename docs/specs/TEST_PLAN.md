# The Analyst - Test Plan (5 Datasets × 20 Queries)

## Test Datasets

### Dataset 1: E-commerce Sales (10K rows)
**Schema**: order_id, customer_id, product_name, order_date, revenue, quantity, category, region
**Characteristics**: Time series, aggregations, filtering, grouping

### Dataset 2: Employee Records (5K rows)  
**Schema**: emp_id, name, department, salary, hire_date, manager_id, location, performance_score
**Characteristics**: Hierarchical data, PII detection, aggregations

### Dataset 3: IoT Sensor Data (50K rows)
**Schema**: sensor_id, timestamp, temperature, humidity, pressure, location, status, battery_level
**Characteristics**: High frequency, time series, anomaly detection queries

### Dataset 4: Financial Transactions (25K rows)
**Schema**: transaction_id, account_id, amount, transaction_date, type, merchant, category, fraud_score
**Characteristics**: Financial data, fraud patterns, time-based analysis

### Dataset 5: Marketing Campaigns (8K rows)
**Schema**: campaign_id, channel, spend, impressions, clicks, conversions, date, target_demographic
**Characteristics**: Marketing metrics, ROI calculations, multi-dimensional analysis

## Canonical Query Categories (20 queries)

### Category 1: Basic Totals (4 queries)
1. "What is the total revenue?"
2. "How many orders do we have?"
3. "What's the average salary by department?"
4. "Total spend across all campaigns"

### Category 2: Group By Aggregations (4 queries)
5. "Revenue by product category"
6. "Sales by region and month"
7. "Employee count by department and location"
8. "Conversion rate by marketing channel"

### Category 3: Time Series (4 queries)
9. "Monthly revenue trends"
10. "Daily transaction volume"
11. "Weekly sensor readings"
12. "Quarterly campaign performance"

### Category 4: Top-N Queries (4 queries)
13. "Top 10 products by revenue"
14. "Highest paid employees"
15. "Most active sensors"
16. "Best performing campaigns"

### Category 5: Filtering & Conditions (4 queries)
17. "Revenue from last 30 days"
18. "Employees with salary > $100k"
19. "Sensors with temperature > 80°F"
20. "Campaigns with ROI > 200%"

## Test Metrics & Success Criteria

### Accuracy Metrics
- **Exact Match**: SQL output matches expected result exactly
- **Tolerance Match**: Numeric results within ±1% tolerance
- **Schema Match**: Generated SQL uses correct table/column names
- **Target**: ≥95% accuracy across all 100 test cases

### Performance Metrics  
- **Response Time**: p95 < 5 seconds for 1M row datasets
- **Query Generation**: < 2 seconds for NL→SQL conversion
- **Validation**: < 500ms for Woodpecker checks
- **Memory Usage**: < 1GB peak during processing

### Security Metrics
- **PII Blocking**: 100% of PII columns blocked from queries
- **Write Prevention**: 0 write queries executed
- **Access Control**: 100% deny-list enforcement
- **Audit Coverage**: Complete trail for all operations

## Test Execution Plan

### Phase 1: Unit Tests (Day 1)
- Schema profiling accuracy
- PII detection precision/recall
- SQL validation logic
- Individual component performance

### Phase 2: Integration Tests (Day 2)
- End-to-end pipeline with synthetic data
- API contract validation
- Error handling scenarios
- Security boundary testing

### Phase 3: Performance Tests (Day 3)
- Large dataset handling (1M+ rows)
- Concurrent user simulation
- Memory leak detection
- Query optimization validation

### Phase 4: User Acceptance (Day 4)
- Pilot user scenarios
- Real-world data testing
- Usability feedback integration
- Edge case handling

## Test Data Management
- **Synthetic Generation**: Use Faker library for consistent test data
- **PII Injection**: Deliberately add PII patterns to test detection
- **Edge Cases**: Empty datasets, malformed CSVs, encoding issues
- **Version Control**: All test datasets versioned and reproducible