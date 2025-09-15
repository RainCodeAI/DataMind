# Step-by-Step Implementation: POST /upload and POST /query

## Phase 1: Foundation Setup (2 hours)

### Step 1.1: Project Structure (30 minutes)
- [ ] Create `/apps/analyst/` directory structure
- [ ] Set up `requirements.txt` with dependencies:
  ```
  fastapi==0.104.1
  pandas==2.1.3
  sqlglot==23.12.0
  openai==1.3.7
  redis==5.0.1
  boto3==1.34.0
  pydantic==2.5.0
  ```
- [ ] Initialize FastAPI app with basic configuration
- [ ] Set up logging configuration with structured format
- [ ] Create environment configuration for API keys and settings

### Step 1.2: Database Models (30 minutes)
- [ ] Create `models.py` with SQLAlchemy models:
  - [ ] `Dataset` model (id, name, status, created_at, user_id)
  - [ ] `ColumnProfile` model (dataset_id, name, type, category, pii_detected)
  - [ ] `AuditEvent` model (event_id, type, timestamp, user_id, details)
- [ ] Set up database connection and migration scripts
- [ ] Create database initialization script

### Step 1.3: Storage Integration (30 minutes)
- [ ] Configure S3-compatible storage for CSV files
- [ ] Set up Redis for caching and job queues
- [ ] Create storage utility functions:
  - [ ] `upload_file_to_storage(file_path, dataset_id)`
  - [ ] `download_file_from_storage(dataset_id)`
  - [ ] `cache_schema(dataset_id, schema_data)`

### Step 1.4: Basic Security (30 minutes)
- [ ] Implement basic authentication middleware
- [ ] Create user session management
- [ ] Set up rate limiting with Redis
- [ ] Add request validation with Pydantic models

## Phase 2: Schema Builder Implementation (3 hours)

### Step 2.1: CSV Processing (1 hour)
- [ ] Implement `semantic_schema.py` core functions:
  - [ ] `SchemaBuilder.__init__()` with PII patterns
  - [ ] `SchemaBuilder.profile_csv()` - main profiling pipeline
  - [ ] `TypeDetector` class with statistical methods
  - [ ] `PIIDetector` class with regex patterns
- [ ] Create CSV validation utilities:
  - [ ] File size validation (100MB limit)
  - [ ] Encoding detection and normalization
  - [ ] Malformed CSV handling
- [ ] Test with sample CSV files

### Step 2.2: Type Detection (1 hour)
- [ ] Implement type detection algorithms:
  - [ ] `detect_numeric_type()` - statistical analysis
  - [ ] `detect_datetime_type()` - pattern matching
  - [ ] `detect_boolean_type()` - value analysis
  - [ ] `detect_string_type()` - fallback classification
- [ ] Add confidence scoring for each type
- [ ] Create unit tests for type detection

### Step 2.3: PII Detection (1 hour)
- [ ] Implement PII detection patterns:
  - [ ] Email regex: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`
  - [ ] Phone regex: `\b\d{3}-\d{3}-\d{4}\b`
  - [ ] SSN regex: `\b\d{3}-\d{2}-\d{4}\b`
  - [ ] Name patterns in column names
- [ ] Add sample data analysis for PII detection
- [ ] Create confidence scoring for PII detection
- [ ] Test with synthetic PII data

## Phase 3: NL→SQL Engine Implementation (3 hours)

### Step 3.1: LLM Integration (1 hour)
- [ ] Implement `nl2sql.py` core functions:
  - [ ] `NL2SQLEngine.__init__()` with LLM client
  - [ ] `NL2SQLEngine.convert_to_sql()` - main conversion pipeline
  - [ ] `generate_sql_with_llm()` - LLM integration
- [ ] Create prompt templates:
  - [ ] System prompt with schema context
  - [ ] Few-shot examples for common queries
  - [ ] Output format specification
- [ ] Test LLM integration with sample queries

### Step 3.2: SQL Validation (1 hour)
- [ ] Implement `SQLValidator` class:
  - [ ] `validate_sql()` - main validation pipeline
  - [ ] `is_select_only()` - security check
  - [ ] `enforce_limit()` - row limit enforcement
  - [ ] `check_blocked_columns()` - PII protection
- [ ] Add sqlglot integration for AST parsing
- [ ] Create validation tests

### Step 3.3: Security Enforcement (1 hour)
- [ ] Implement `SecurityEnforcer` class:
  - [ ] `check_column_access()` - column-level security
  - [ ] `check_operation_allowed()` - operation whitelist
  - [ ] `sanitize_query()` - query sanitization
- [ ] Add query complexity scoring
- [ ] Create security test suite

## Phase 4: API Implementation (2 hours)

### Step 4.1: POST /upload Endpoint (1 hour)
- [ ] Create FastAPI endpoint:
  ```python
  @app.post("/upload", response_model=UploadResponse)
  async def upload_dataset(file: UploadFile, dataset_name: str)
  ```
- [ ] Implement file handling:
  - [ ] Multipart form data parsing
  - [ ] File validation (size, type, encoding)
  - [ ] Temporary file storage
- [ ] Add background profiling job:
  - [ ] Queue profiling task with Redis
  - [ ] Return immediate response with dataset_id
  - [ ] Update status via WebSocket or polling
- [ ] Add error handling and validation

### Step 4.2: POST /query Endpoint (1 hour)
- [ ] Create FastAPI endpoint:
  ```python
  @app.post("/query", response_model=QueryResponse)
  async def execute_query(request: QueryRequest)
  ```
- [ ] Implement query pipeline:
  - [ ] Validate dataset exists and is ready
  - [ ] Call NL→SQL engine
  - [ ] Execute SQL against dataset
  - [ ] Generate chart suggestions
  - [ ] Log audit event
- [ ] Add response formatting and error handling

## Phase 5: Testing & Integration (1 hour)

### Step 5.1: Unit Tests (30 minutes)
- [ ] Test schema profiling with sample datasets
- [ ] Test NL→SQL conversion with canonical queries
- [ ] Test security validation with malicious inputs
- [ ] Test API endpoints with various inputs

### Step 5.2: Integration Tests (30 minutes)
- [ ] End-to-end upload and query workflow
- [ ] Error handling and edge cases
- [ ] Performance testing with large files
- [ ] Security penetration testing

## Success Criteria Checklist

### Technical Requirements
- [ ] POST /upload accepts CSV files up to 100MB
- [ ] Schema profiling completes within 30 seconds
- [ ] POST /query generates valid SQL for test queries
- [ ] All security constraints enforced (SELECT-only, LIMIT 1000)
- [ ] PII detection blocks sensitive columns
- [ ] Audit logging captures all operations

### Performance Targets
- [ ] Upload processing: < 30 seconds for 1M rows
- [ ] Query generation: < 2 seconds
- [ ] Total query pipeline: < 5 seconds p95
- [ ] Memory usage: < 1GB peak

### Security Validation
- [ ] 100% PII column blocking
- [ ] 0 write queries executed
- [ ] Complete audit trail
- [ ] Rate limiting functional

## Next Steps After Implementation
1. **Deploy to staging** environment
2. **Run full test suite** with 5 datasets × 20 queries
3. **Performance benchmarking** with production-scale data
4. **Security audit** and penetration testing
5. **Pilot user onboarding** preparation

## Rollback Plan
- [ ] Database migration rollback scripts
- [ ] Feature flags for gradual rollout
- [ ] Monitoring alerts for critical failures
- [ ] Incident response procedures
