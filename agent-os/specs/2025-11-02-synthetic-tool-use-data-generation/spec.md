# Specification: Synthetic Tool-Use Data Generation

## Goal
Generate 1,000-3,000 OpenAI-style function calling conversation examples using Claude Haiku 4.5 API to create realistic weather tool-use training data with persona integration for LoRA fine-tuning.

## User Stories
- As an ML engineer, I want automated synthetic data generation so I can create diverse tool-calling examples without manual effort
- As a model trainer, I want both persona-styled and neutral tool-use examples so the model learns robust function calling while maintaining literary character

## Specific Requirements

**Claude Haiku 4.5 API Integration**
- Use Anthropic Python SDK to call Claude Haiku 4.5 for conversation generation
- Prompt user for Anthropic API key at script startup if not found in environment
- Implement retry logic with exponential backoff for API failures or rate limits
- Include request throttling to respect API rate limits
- Log API call metrics (total calls, tokens used, failures) for cost tracking
- Use streaming disabled for full response validation before processing

**Weather Tool Schema Design**
- Model tool schema on Open-Meteo API endpoints (current weather, forecast, geocoding)
- Define get_current_weather(latitude, longitude) function returning temperature, conditions, wind
- Define get_forecast(latitude, longitude, days) function returning multi-day forecast data
- Define geocode_location(city, country) helper function converting city names to coordinates
- Include realistic parameter validation (latitude -90 to 90, longitude -180 to 180, days 1-14)
- Generate semantically realistic weather responses (temperature ranges appropriate for location/season, plausible condition combinations)

**Scenario Distribution and Coverage**
- Generate 60-70% success cases with valid tool calls and realistic weather responses
- Generate 15-20% error handling cases (invalid locations, missing parameters, out-of-range values, API errors)
- Generate 15-20% multi-turn conversations (follow-up questions, clarifications, multiple location queries)
- Ensure geographic diversity (50+ different cities across continents, various climate zones)
- Cover edge cases: extreme weather, null islands, timezone differences, unit conversions (Celsius/Fahrenheit)

**Persona Integration Strategy**
- Generate 60% neutral-tone examples (professional assistant, focus on accurate tool calling mechanics)
- Generate 25% Twain-style examples (witty, humorous responses using weather data with literary flair)
- Generate 15% Franklin-style examples (didactic, almanac-style wisdom incorporating weather observations)
- Apply persona only to assistant responses AFTER receiving tool results, not to tool call structure
- Ensure tool call JSON remains identical across personas (only response prose differs)

**JSONL Output Format and Metadata**
- Write output to data/synthetic/tool_use_examples.jsonl using existing output_writer.py atomic write pattern
- Use OpenAI chat format: messages array with role fields (system, user, assistant, tool)
- Include tool_calls array in assistant messages with function name and JSON arguments
- Include tool role messages with tool_call_id and stringified JSON response content
- Add tags metadata: persona (twain/franklin/neutral), tone (humorous/didactic/neutral), domain (weather, tool_use), source (synthetic)
- Generate unique conversation IDs for tracking and debugging

**Validation and Quality Assurance**
- Extend chat_format_validator.py to validate tool_calls schema (function name, arguments structure)
- Validate all tool call JSON arguments parse correctly and match function schemas
- Perform semantic validation on weather data (temperatures in reasonable ranges for locations, valid condition codes)
- Check that assistant responses reference tool output data (groundedness validation)
- Validate role ordering including tool messages (system, user, assistant with tool_calls, tool, assistant)
- Generate validation report with pass/fail counts, error details, and sample rejected examples

**Automated Execution and Progress Tracking**
- Implement single-script execution: python scripts/generate_synthetic_tool_data.py
- Display progress bar showing generation status (X/1000 examples, Y% complete)
- Log batch statistics every 100 examples (success rate, average response length, API latency)
- Print final summary: total examples, persona distribution, scenario type breakdown, validation pass rate
- Save generation metadata to data/synthetic/generation_metadata.json (timestamp, API version, config parameters)

**Error Handling and Resilience**
- Catch and log API errors without stopping entire generation process
- Implement regeneration for invalid outputs (max 3 retries per example before skipping)
- Save intermediate checkpoints every 250 examples to allow resumption on failure
- Gracefully handle keyboard interrupt (Ctrl+C) with option to save partial results
- Validate final output file integrity and log any corruption warnings

## Out of Scope
- Hand-crafted or template-based example generation (must use LLM)
- Manual review checkpoints during generation process
- Multi-API support (only Open-Meteo weather schema, not OpenWeatherMap or others)
- Real API calls to weather services (use mock/synthetic responses)
- Integration with other data pipelines or automatic merging with existing datasets
- Training execution or model evaluation (dataset creation only)
- Web scraping for real weather data examples
- Interactive mode or GUI for generation control
- Multi-language support (English only)
- Custom tool schemas beyond weather domain
