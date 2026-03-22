# University Programs Classification Example

Classifies university program names into 12 public, high-level academic categories using an LLM.

## Quick Start

1. Set your API key:
   ```bash
   export LLM_API_KEY=your-deepseek-api-key
   ```

2. Run classification:
   ```bash
   llm-classify run --config examples/university-programs/classify.yaml
   ```

3. Results will be in the `output/` directory.

## Files

- `classify.yaml` — Configuration with 12 public demo categories and rate limiting
- `prompt.txt` — Chinese prompt template for the LLM
- `sample_input.csv` — 20 sample university programs
