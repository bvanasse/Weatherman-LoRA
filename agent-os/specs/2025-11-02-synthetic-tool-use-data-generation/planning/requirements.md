# Spec Requirements: Synthetic Tool-Use Data Generation

## Initial Description
Synthetic Tool-Use Data Generation

## Requirements Discussion

### First Round Questions

**Q1:** I assume we'll generate synthetic conversations using an LLM (like GPT-4 or Claude) to create the tool-call examples, rather than hand-crafting them. Is that correct, or do you prefer a template-based generation approach?

**Answer:** Yes I would prefer to use something like Claude Haiku 4.5 to generate the synthetic conversations and create the tool-call examples.

**Q2:** For the weather API tool schema, I'm thinking we should use **Open-Meteo API** as specified in the tech stack (free, no key required). Should we model the tool schema exactly on Open-Meteo's endpoints, or create a simplified/abstracted weather tool interface?

**Answer:** Use your best judgement.

**Q3:** I assume the 1,000-3,000 examples should cover:
- **Success cases** (valid tool calls with realistic weather responses)
- **Error handling** (invalid locations, missing parameters, API errors)
- **Multi-turn conversations** (follow-up questions about weather)

Should we target roughly equal distribution, or prioritize success cases (e.g., 70% success, 20% multi-turn, 10% errors)?

**Answer:** Yes. Correct. Include more examples if necessary.

**Q4:** For the persona integration, should the **assistant's responses** (after receiving tool results) maintain the Twain/Franklin literary style, or should tool-use examples use a more neutral tone to avoid conflating style with tool-calling mechanics?

**Answer:** Use persona integration, why not - could be interesting. Maybe still include a portion of more neutral examples as well to allow for better tool calling mechanics.

**Q5:** I'm thinking the output format should be **JSONL with OpenAI chat format** including `tool_calls` and `tool` role messages, matching the schema in tech-stack.md. Should we also include the **tags metadata** (persona/tone/domain/source) for each example?

**Answer:** Yes.

**Q6:** For validation, I assume we need **JSON schema validation** to ensure all function calls are syntactically correct. Should we also validate that tool responses are **semantically realistic** (e.g., temperatures in reasonable ranges for locations)?

**Answer:** Yes

**Q7:** Should the generation process be **fully automated** (single script execution), or do you want **intermediate checkpoints** where we can review samples before generating the full 1,000-3,000 examples?

**Answer:** Fully automated. Prompt the user for API keys if necessary.

**Q8:** Are there any **specific edge cases or scenarios** we should exclude from the synthetic data? For example, should we avoid certain types of weather queries, locations, or conversational patterns?

**Answer:** None come to mind. Do your best work to achieve the mission.

### Existing Code to Reference

No similar existing features identified for reference.

### Follow-up Questions

No follow-up questions required.

## Visual Assets

### Files Provided:
No visual assets provided.

### Visual Insights:
N/A

## Requirements Summary

### Functional Requirements

**Core Functionality:**
- Generate 1,000-3,000 synthetic tool-use conversation examples using Claude Haiku 4.5
- Create OpenAI-style function calling examples with weather API tool schema
- Cover three primary scenario types:
  - Success cases: Valid tool calls with realistic weather responses
  - Error handling: Invalid locations, missing parameters, API errors
  - Multi-turn conversations: Follow-up questions about weather data
- Target flexible distribution with more examples if needed to ensure comprehensive coverage

**LLM Generation:**
- Use Claude Haiku 4.5 API for automated synthetic conversation generation
- Prompt the user for Anthropic API keys during script execution
- Fully automated process (single script execution, no manual checkpoints)

**Weather API Integration:**
- Model tool schema on Open-Meteo API (free, no key required)
- Use best judgment for exact schema design vs. abstracted interface
- Generate realistic weather API responses for tool call results

**Persona Integration:**
- Include literary persona (Twain/Franklin style) in assistant responses after tool calls
- Also generate neutral-tone examples to ensure robust tool-calling mechanics
- Mixed distribution to balance personality with functional tool-use capability

**Output Format:**
- JSONL (JSON Lines) format with one example per line
- OpenAI chat format with `role` fields: `system`, `user`, `assistant`, `tool`
- Include `tool_calls` structure in assistant messages
- Include `tool` role messages with API response content
- Add tags metadata for each example: `persona`, `tone`, `domain`, `source`

**Validation:**
- JSON schema validation: Ensure all function calls are syntactically correct
- Semantic validation: Verify tool responses are realistic (e.g., temperatures in reasonable ranges for locations)
- Automated quality checks before finalizing dataset

### Reusability Opportunities

No existing components identified for reuse. This is a new capability for the project.

### Scope Boundaries

**In Scope:**
- Automated generation of 1,000-3,000 tool-use examples
- Claude Haiku 4.5 integration for synthetic data creation
- Weather API tool schema design (Open-Meteo based)
- Success cases, error handling, and multi-turn conversation coverage
- Persona-integrated and neutral-tone examples
- JSONL output with OpenAI chat format and metadata tags
- JSON schema and semantic validation
- Automated execution with API key prompting

**Out of Scope:**
- Hand-crafted or template-based generation (LLM-based only)
- Manual review checkpoints during generation
- Multi-API support (focus on Open-Meteo for weather)
- Integration with other data pipelines (this is standalone dataset creation)
- Training execution (that's roadmap item #10)

### Technical Considerations

**API Requirements:**
- Anthropic API access for Claude Haiku 4.5
- API key collection via user prompt during script execution
- Rate limiting and error handling for API calls

**Data Quality:**
- Ensure diverse geographic locations in weather queries
- Realistic temperature ranges, conditions, and weather patterns
- Valid JSON structure for all tool calls and responses
- Proper error messages for failure scenarios

**Integration Points:**
- Output must be compatible with existing training data format (JSONL with messages and tags)
- Should merge seamlessly with literary style and humor datasets in roadmap item #10
- Follows tech stack data format standards (OpenAI-style tool calls)

**Technology Stack:**
- Python 3.10+ for scripting
- Anthropic Python SDK for Claude Haiku 4.5 API
- Open-Meteo API documentation for schema design
- jsonlines library for JSONL output
- JSON schema validation libraries (jsonschema or similar)

**Performance Goals:**
- Generate 1,000-3,000 examples efficiently
- Automated validation to catch errors early
- Clear progress indicators during generation
- Final dataset ready for immediate use in training pipeline

**Edge Cases to Handle:**
- API failures during generation (retry logic)
- Invalid JSON from LLM (validation and regeneration)
- Unrealistic weather data (semantic validation filters)
- Rate limiting from Anthropic API (throttling logic)
