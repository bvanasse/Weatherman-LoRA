# Task Breakdown: Synthetic Tool-Use Data Generation

## Overview
Total Tasks: 5 Task Groups
Target Output: 1,000-3,000 synthetic tool-use conversation examples in JSONL format

## Task List

### Tool Schema & Data Modeling

#### Task Group 1: Weather Tool Schema Design and Mock Response Generation
**Dependencies:** None

- [x] 1.0 Complete tool schema and mock data infrastructure
  - [x] 1.1 Write 2-8 focused tests for tool schema and mock responses
    - Test get_current_weather schema validation (latitude/longitude bounds, required fields)
    - Test get_forecast schema validation (days parameter 1-14 range)
    - Test geocode_location mock responses (valid city/country mapping)
    - Test semantic weather data realism (temperature ranges for climate zones)
    - Test error response generation (invalid parameters, missing data)
  - [x] 1.2 Define weather tool function schemas
    - Create get_current_weather(latitude, longitude) schema with parameter types and descriptions
    - Create get_forecast(latitude, longitude, days) schema with parameter validation rules
    - Create geocode_location(city, country) helper schema for coordinate lookup
    - Define JSON response structures matching OpenAI function calling format
  - [x] 1.3 Implement geographic location database
    - Create dataset of 50+ diverse cities with coordinates across continents
    - Include climate zone metadata (tropical, temperate, arctic, desert, etc.)
    - Add seasonal temperature ranges for semantic validation
    - Include edge cases (null island, extreme latitudes, timezone variations)
  - [x] 1.4 Build mock weather response generator
    - Generate realistic current weather responses (temperature, conditions, wind, humidity)
    - Generate realistic forecast responses (multi-day temperature/condition arrays)
    - Apply climate-appropriate ranges (e.g., Phoenix: 70-110°F summer, Moscow: -20-30°F winter)
    - Include weather condition variety (clear, cloudy, rain, snow, thunderstorm, fog)
  - [x] 1.5 Create error response generator
    - Invalid location errors (out-of-bounds coordinates, unknown cities)
    - Missing parameter errors (latitude/longitude/days not provided)
    - Out-of-range errors (days > 14, latitude > 90)
    - API error simulation (rate limit, service unavailable)
  - [x] 1.6 Ensure tool schema tests pass
    - Run ONLY the 2-8 tests written in 1.1
    - Verify schema validation catches invalid parameters
    - Verify mock responses are semantically realistic

**Acceptance Criteria:**
- The 2-8 tests written in 1.1 pass
- Tool schemas match OpenAI function calling format
- Mock responses generate realistic weather data for 50+ cities
- Error responses cover key failure scenarios

### API Integration Layer

#### Task Group 2: Claude Haiku 4.5 API Integration and Prompt Engineering
**Dependencies:** Task Group 1

- [x] 2.0 Complete Claude API integration
  - [x] 2.1 Write 2-8 focused tests for API integration
    - Test API client initialization with valid/invalid keys
    - Test retry logic with exponential backoff on failures
    - Test rate limiting throttle mechanism
    - Test API response parsing and extraction
    - Test streaming disabled configuration
  - [x] 2.2 Implement Anthropic API client wrapper
    - Install and configure Anthropic Python SDK
    - Create API key handling (prompt user if not in environment variable ANTHROPIC_API_KEY)
    - Implement retry logic with exponential backoff (max 3 retries, 1s/2s/4s delays)
    - Add rate limiting throttle (respect Anthropic API limits)
    - Log API metrics (total calls, tokens used, latency, failures)
  - [x] 2.3 Design conversation generation prompts
    - Create system prompt for success case generation (neutral, Twain, Franklin personas)
    - Create system prompt for error handling scenarios
    - Create system prompt for multi-turn conversations
    - Include tool schema definitions in prompts
    - Specify exact output format (OpenAI tool_calls structure)
  - [x] 2.4 Build prompt template engine
    - Create template for single-turn success cases with persona injection
    - Create template for multi-turn conversations (2-3 exchanges)
    - Create template for error scenarios with appropriate user queries
    - Inject location/weather context dynamically from location database
  - [x] 2.5 Implement conversation parser
    - Parse Claude API responses into structured conversation objects
    - Extract assistant messages with tool_calls arrays
    - Validate tool_call JSON structure (function name, arguments)
    - Handle malformed responses with regeneration logic
  - [x] 2.6 Ensure API integration tests pass
    - Run ONLY the 2-8 tests written in 2.1
    - Verify API client handles retries correctly
    - Verify rate limiting prevents quota exhaustion

**Acceptance Criteria:**
- The 2-8 tests written in 2.1 pass
- API client successfully authenticates and calls Claude Haiku 4.5
- Prompts generate tool-calling conversations in correct format
- Retry and rate limiting work correctly

### Conversation Generation Engine

#### Task Group 3: Synthetic Conversation Generation Pipeline
**Dependencies:** Task Groups 1-2

- [x] 3.0 Complete conversation generation pipeline
  - [x] 3.1 Write 2-8 focused tests for generation pipeline
    - Test scenario distribution (60-70% success, 15-20% error, 15-20% multi-turn)
    - Test persona distribution (60% neutral, 25% Twain, 15% Franklin)
    - Test conversation structure (system, user, assistant with tool_calls, tool, assistant)
    - Test unique conversation ID generation
    - Test metadata tag assignment (persona, tone, domain, source)
  - [x] 3.2 Build conversation orchestrator
    - Implement scenario type selection logic (success/error/multi-turn distribution)
    - Implement persona selection logic (neutral/Twain/Franklin distribution)
    - Coordinate location selection from geographic database
    - Track generation statistics (scenarios generated, personas used, locations covered)
  - [x] 3.3 Create conversation assembly pipeline
    - Generate system message with persona instruction
    - Generate user query based on scenario type and location
    - Call Claude API to generate assistant response with tool_calls
    - Insert mock tool response message with tool_call_id
    - Generate final assistant response using tool data
  - [x] 3.4 Implement persona-specific response generation
    - Apply neutral tone (60%): professional, accurate weather interpretation
    - Apply Twain persona (25%): witty, humorous observations about weather
    - Apply Franklin persona (15%): didactic, almanac-style wisdom from weather data
    - Ensure tool_calls JSON structure identical across personas
  - [x] 3.5 Add conversation metadata and tagging
    - Generate unique conversation IDs (UUID format)
    - Add tags: persona (twain/franklin/neutral), tone (humorous/didactic/neutral)
    - Add tags: domain (weather, tool_use), source (synthetic)
    - Include generation timestamp and Claude API version
  - [x] 3.6 Ensure generation pipeline tests pass
    - Run ONLY the 2-8 tests written in 3.1
    - Verify scenario and persona distributions match targets
    - Verify conversation structure follows OpenAI format

**Acceptance Criteria:**
- The 2-8 tests written in 3.1 pass
- Conversations follow correct OpenAI tool-calling format
- Scenario and persona distributions match spec requirements
- Geographic diversity achieved (50+ cities)

### Validation & Quality Assurance

#### Task Group 4: Validation Pipeline and Quality Checks
**Dependencies:** Task Groups 1-3

- [x] 4.0 Complete validation and quality assurance pipeline
  - [x] 4.1 Write 2-8 focused tests for validation logic
    - Test tool_calls schema validation (function name, arguments structure)
    - Test JSON argument parsing and schema compliance
    - Test semantic weather data validation (temperature ranges, valid conditions)
    - Test groundedness check (assistant references tool output)
    - Test role ordering validation (system, user, assistant, tool, assistant)
  - [x] 4.2 Extend chat_format_validator.py for tool-calling
    - Add validation for tool_calls array in assistant messages
    - Add validation for function name matching defined schemas
    - Add validation for arguments JSON structure and types
    - Add validation for tool role messages with tool_call_id
    - Update role_order validation to support tool role
  - [x] 4.3 Implement JSON schema validator
    - Validate all tool_call arguments parse as valid JSON
    - Validate arguments match function parameter schemas (latitude/longitude/days types)
    - Validate parameter ranges (latitude -90 to 90, longitude -180 to 180, days 1-14)
    - Detect and log malformed JSON with context
  - [x] 4.4 Build semantic validation engine
    - Validate temperature values reasonable for location climate zone
    - Validate weather conditions exist in defined set (clear, cloudy, rain, etc.)
    - Validate forecast day counts match request
    - Flag unrealistic combinations (e.g., snow in tropical region in July)
  - [x] 4.5 Create groundedness validator
    - Parse assistant final response for references to tool output data
    - Check that mentioned temperatures/conditions match tool response
    - Detect hallucinated weather data not in tool output
    - Score groundedness (0-100% data usage from tool)
  - [x] 4.6 Implement validation reporting
    - Track validation pass/fail counts per check type
    - Log sample rejected examples with failure reasons
    - Generate validation summary report (overall pass rate, common failures)
    - Save detailed validation log for debugging
  - [x] 4.7 Ensure validation tests pass
    - Run ONLY the 2-8 tests written in 4.1
    - Verify schema validation catches malformed tool_calls
    - Verify semantic validation flags unrealistic data

**Acceptance Criteria:**
- The 2-8 tests written in 4.1 pass
- Extended chat_format_validator.py handles tool messages
- Validation catches JSON, semantic, and groundedness issues
- Validation report provides actionable quality metrics

### Orchestration & Execution

#### Task Group 5: Main Script, Progress Tracking, and Error Handling
**Dependencies:** Task Groups 1-4

- [x] 5.0 Complete main generation script and orchestration
  - [x] 5.1 Write 2-8 focused tests for orchestration logic
    - Test main script execution flow (initialization, generation loop, finalization)
    - Test progress tracking and statistics logging
    - Test checkpoint saving every 250 examples
    - Test graceful keyboard interrupt handling (Ctrl+C)
    - Test regeneration logic (max 3 retries before skip)
  - [x] 5.2 Create main generation script: scripts/generate_synthetic_tool_data.py
    - Implement single-entry-point execution with clear usage instructions
    - Add command-line arguments (--count, --output-path, --checkpoint-dir)
    - Initialize all components (API client, schema, orchestrator, validators)
    - Implement main generation loop (1,000-3,000 examples)
  - [x] 5.3 Implement progress tracking and logging
    - Display progress bar (X/N examples, Y% complete) using tqdm library
    - Log batch statistics every 100 examples (success rate, avg response length, API latency)
    - Print scenario/persona distribution updates
    - Track and display validation pass rate
  - [x] 5.4 Build checkpoint and resumption system
    - Save intermediate checkpoints every 250 examples to data/synthetic/checkpoints/
    - Store generation state (examples generated, RNG seed, statistics)
    - Implement resumption from checkpoint (detect partial run, offer to continue)
    - Clean up checkpoint files on successful completion
  - [x] 5.5 Implement error handling and resilience
    - Catch API errors without stopping generation (log and continue)
    - Implement regeneration on validation failure (max 3 retries per example)
    - Handle keyboard interrupt (Ctrl+C) gracefully with save prompt
    - Validate final output file integrity (JSONL parse check)
  - [x] 5.6 Add final output and reporting
    - Write validated examples to data/synthetic/tool_use_examples.jsonl using output_writer.py
    - Save generation metadata to data/synthetic/generation_metadata.json
    - Print final summary (total examples, persona breakdown, scenario breakdown, validation pass rate)
    - Generate timestamp and API version in metadata
  - [x] 5.7 Ensure orchestration tests pass
    - Run ONLY the 2-8 tests written in 5.1
    - Verify checkpoint save/resume works correctly
    - Verify error handling prevents crashes

**Acceptance Criteria:**
- The 2-8 tests written in 5.1 pass
- Main script generates 1,000-3,000 examples successfully
- Progress tracking provides clear visibility
- Checkpoints and error handling ensure resilience

### Integration Testing & Verification

#### Task Group 6: End-to-End Testing and Dataset Quality Review
**Dependencies:** Task Groups 1-5

- [x] 6.0 Complete integration testing and quality verification
  - [x] 6.1 Review tests from Task Groups 1-5
    - Review the 2-8 tests from Task Group 1 (tool schema)
    - Review the 2-8 tests from Task Group 2 (API integration)
    - Review the 2-8 tests from Task Group 3 (generation pipeline)
    - Review the 2-8 tests from Task Group 4 (validation)
    - Review the 2-8 tests from Task Group 5 (orchestration)
    - Total existing tests: approximately 10-40 tests
  - [x] 6.2 Analyze test coverage gaps for THIS feature only
    - Identify critical end-to-end workflows lacking coverage
    - Focus on integration between components (API → generation → validation)
    - Check coverage for edge cases in spec (extreme weather, multi-turn, error handling)
    - Prioritize dataset quality verification over unit test gaps
  - [x] 6.3 Write up to 10 additional strategic tests maximum
    - Add end-to-end test: generate single example from API call to validated JSONL output
    - Add integration test: verify persona distribution across 100-example batch
    - Add integration test: verify geographic diversity (50+ unique cities in dataset)
    - Add integration test: verify scenario distribution (success/error/multi-turn ratios)
    - Add quality test: validate sample conversations for groundedness and realism
    - Add resilience test: verify recovery from API failure mid-generation
    - Skip exhaustive edge case testing (focus on critical user workflows)
  - [x] 6.4 Run feature-specific tests only
    - Run ONLY tests related to this feature (tests from 1.1, 2.1, 3.1, 4.1, 5.1, and 6.3)
    - Expected total: approximately 20-50 tests maximum
    - Do NOT run entire application test suite
    - Verify all critical workflows pass
  - [x] 6.5 Execute full dataset generation and quality review
    - Run: python scripts/generate_synthetic_tool_data.py --count 1000
    - Monitor progress and statistics during generation
    - Review final validation report for quality metrics
    - Manually inspect 10-20 random examples for quality (realism, persona accuracy, groundedness)
  - [x] 6.6 Verify output compatibility with training pipeline
    - Validate JSONL format matches existing training data schema (from previous specs)
    - Confirm tags structure compatible with instructionalization pipeline
    - Test that output_writer.py atomic write worked correctly
    - Verify file can be loaded by data_loader.py (from normalization pipeline)
  - [x] 6.7 Create usage documentation
    - Write README section: scripts/README_SYNTHETIC_TOOL_DATA.md
    - Document script usage: python scripts/generate_synthetic_tool_data.py --help
    - Document output format and metadata fields
    - Provide example commands for different generation scenarios
    - Document checkpoint resumption process

**Acceptance Criteria:**
- All feature-specific tests pass (approximately 20-50 tests total)
- Generated dataset meets quality standards (1,000+ examples, correct distributions)
- Output compatible with existing training pipeline
- No more than 10 additional tests added
- Documentation enables easy usage by other engineers

## Execution Order

Recommended implementation sequence:
1. **Tool Schema & Data Modeling** (Task Group 1) - Foundation for mock weather data
2. **API Integration Layer** (Task Group 2) - Claude Haiku 4.5 API setup and prompting
3. **Conversation Generation Engine** (Task Group 3) - Core generation logic with personas
4. **Validation & Quality Assurance** (Task Group 4) - Quality checks and validators
5. **Orchestration & Execution** (Task Group 5) - Main script, progress, error handling
6. **Integration Testing & Verification** (Task Group 6) - End-to-end testing and quality review

## Notes

- **API Costs**: Generating 1,000-3,000 examples with Claude Haiku 4.5 will incur API costs. Monitor token usage.
- **Persona Quality**: Manual review of sample outputs recommended to ensure persona styles are appropriate and distinct.
- **Reusability**: Leverages existing output_writer.py, chat_format_validator.py, and config_loader.py patterns.
- **Extensibility**: Tool schema design allows future addition of other weather functions (alerts, historical data).
