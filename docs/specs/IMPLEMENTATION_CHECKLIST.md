# The Analyst - Implementation Checklist (Jira/Trello Ready)

## Epic: Schema Builder v1 → Constrained NL→SQL Thin Slice

### Phase 1: Foundation & Infrastructure (Week 1, Days 1-2)

#### 🏗️ Backend Infrastructure
- [ ] Set up FastAPI application structure in `/apps/analyst/`
- [ ] Configure async CSV processing pipeline
- [ ] Implement S3-compatible storage integration
- [ ] Set up Redis for caching and job queues
- [ ] Create database models for datasets and audit logs
- [ ] Implement basic authentication and RBAC structure

#### 📊 Schema Profiling Engine
- [ ] Build CSV validation and sanitization module
- [ ] Implement data type detection (string, numeric, datetime, boolean)
- [ ] Create category classification (identifier, metric, temporal, categorical)
- [ ] Add unit detection and synonym mapping
- [ ] Build PII detection pipeline with regex patterns
- [ ] Create semantic schema generation from profiling results

### Phase 2: Core NL→SQL Engine (Week 1, Days 3-4)

#### 🔒 Security & Validation
- [ ] Integrate sqlglot for SQL parsing and validation
- [ ] Implement SELECT-only query enforcement
- [ ] Add mandatory LIMIT 1000 constraint
- [ ] Create column deny-list enforcement
- [ ] Build query complexity scoring and limits
- [ ] Implement Interceptor prompt sanitization

#### 🤖 LLM Integration
- [ ] Set up LLM API integration (OpenAI/Anthropic)
- [ ] Create prompt templates for NL→SQL conversion
- [ ] Implement context-aware query generation
- [ ] Add confidence scoring for generated queries
- [ ] Build query explanation and reasoning output

### Phase 3: Validation & Governance (Week 1, Days 5-6)

#### 🔍 Woodpecker Integration
- [ ] Implement post-query validation pipeline
- [ ] Add schema consistency checking
- [ ] Create aggregation safety validation
- [ ] Build row count estimation and limits
- [ ] Implement performance cost estimation
- [ ] Add result sanity checking

#### 📈 Chart Generation
- [ ] Create chart type suggestion engine
- [ ] Implement basic chart configurations (bar, line, scatter)
- [ ] Add chart data validation and formatting
- [ ] Build chart preview generation
- [ ] Create chart export capabilities

### Phase 4: API & Integration (Week 1, Days 7-8)

#### 🌐 REST API Implementation
- [ ] Build POST /upload endpoint with file handling
- [ ] Implement GET /profile/{dataset_id} with semantic schema
- [ ] Create POST /query endpoint with full pipeline
- [ ] Add error handling and status codes
- [ ] Implement rate limiting and request validation
- [ ] Create API documentation with OpenAPI/Swagger

#### 🔗 Corral Integration
- [ ] Integrate Interceptor for prompt cleaning
- [ ] Connect Woodpecker for output validation
- [ ] Set up Charlie for export governance (optional)
- [ ] Implement audit logging throughout pipeline
- [ ] Add compliance reporting capabilities

### Phase 5: Testing & Quality Assurance (Week 2, Days 1-2)

#### 🧪 Test Suite Implementation
- [ ] Create 5 synthetic test datasets
- [ ] Implement 20 canonical query test cases
- [ ] Build automated accuracy testing framework
- [ ] Add performance benchmarking suite
- [ ] Create security penetration testing
- [ ] Implement integration test suite

#### 📊 Monitoring & Observability
- [ ] Set up application performance monitoring
- [ ] Implement structured logging throughout
- [ ] Create metrics collection (accuracy, latency, errors)
- [ ] Build alerting for critical failures
- [ ] Add health check endpoints
- [ ] Implement audit trail validation

### Phase 6: Pilot Preparation (Week 2, Days 3-5)

#### 🚀 Deployment & Configuration
- [ ] Set up staging environment
- [ ] Configure production deployment pipeline
- [ ] Implement feature flags for gradual rollout
- [ ] Create user onboarding documentation
- [ ] Set up monitoring dashboards
- [ ] Prepare pilot user training materials

#### 👥 User Experience
- [ ] Build basic Streamlit UI for testing
- [ ] Create dataset upload interface
- [ ] Implement query input and results display
- [ ] Add chart visualization components
- [ ] Create user feedback collection system
- [ ] Build help documentation and examples

## Success Criteria Checklist

### Technical Metrics
- [ ] ≥95% accuracy on 100 test query suite
- [ ] p95 response time < 5 seconds for 1M row datasets
- [ ] 100% PII deny-list enforcement
- [ ] 0 write queries executed (security validation)
- [ ] Complete audit trail for all operations

### User Acceptance
- [ ] 2-3 Stratford businesses successfully onboarded
- [ ] Positive feedback on query accuracy and usability
- [ ] No critical security or data privacy issues
- [ ] Successful integration with pilot business data

### Business Readiness
- [ ] Documentation complete for pilot users
- [ ] Support processes established
- [ ] Performance monitoring operational
- [ ] Compliance reporting functional
- [ ] Next sprint planning completed

## Risk Mitigation Tasks

### High Priority
- [ ] Implement comprehensive PII detection testing
- [ ] Set up automated security scanning
- [ ] Create performance regression testing
- [ ] Build disaster recovery procedures
- [ ] Establish incident response playbook

### Medium Priority  
- [ ] Create user training materials
- [ ] Set up feedback collection system
- [ ] Implement gradual feature rollout
- [ ] Build monitoring dashboards
- [ ] Create troubleshooting documentation