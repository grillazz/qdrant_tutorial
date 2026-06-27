/Users/waco/.local/bin/uv run /Users/waco/projects/2026/qdrant_poc/.venv/bin/python /Users/waco/projects/2026/qdrant_poc/semantic_jira_search.py 
Loading weights: 100%|██████████| 199/199 [00:00<00:00, 8812.31it/s]
Loading weights: 100%|██████████| 199/199 [00:00<00:00, 6441.62it/s]
Uploaded 115 vectors across three chunking strategies
Query: 'keeping plants from drying out and avoiding root rot'

--- FIXED CHUNKING ---
1. GARDEN-019 (Replace damaged mulch around blueberry bushes) | Score: 0.509
   Chunk: consistent soil moisture, moderates temperature extremes, and slowly adds organic matter back into t...
2. GARDEN-012 (Remove mildew from squash leaves) | Score: 0.474
   Chunk: . managing mildew early helps preserve photosynthetic leaf area, supports fruit development, and red...
3. GARDEN-015 (Plant lettuce succession crop) | Score: 0.465
   Chunk: gaps left by earlier harvests and making efficient use of available space. keeping the soil cool and...

--- SENTENCE CHUNKING ---
1. GARDEN-010 (Check soil moisture in peach orchard) | Score: 0.577
   Chunk: Maintaining proper soil moisture is crucial for fruit set, fruit size, and overall tree health, espe...
2. GARDEN-012 (Remove mildew from squash leaves) | Score: 0.494
   Chunk: Monitor neighboring plants closely, as mildew can spread rapidly under warm, humid conditions. Manag...
3. GARDEN-017 (Water newly planted pepper seedlings) | Score: 0.488
   Chunk: As the plants grow, gradually increase the volume of water to match their expanding root system. Ade...

--- SEMANTIC CHUNKING ---
1. GARDEN-010 (Check soil moisture in peach orchard) | Score: 0.574
   Chunk: If soil is drying too quickly, adjust the irrigation schedule or flow rate to maintain consistent mo...
2. GARDEN-017 (Water newly planted pepper seedlings) | Score: 0.488
   Chunk: As the plants grow, gradually increase the volume of water to match their expanding root system. Ade...
3. GARDEN-015 (Plant lettuce succession crop) | Score: 0.436
   Chunk: Mark the area with a label indicating the variety and sowing date so that future maintenance and har...

CHUNKING STRATEGY ANALYSIS
========================================

FIXED STRATEGY:
  Total chunks: 38
  Avg chunk size: 357 chars
  Size range: 112-592 chars

SENTENCE STRATEGY:
  Total chunks: 39
  Avg chunk size: 360 chars
  Size range: 162-538 chars

SEMANTIC STRATEGY:
  Total chunks: 38
  Avg chunk size: 356 chars
  Size range: 163-568 chars

Process finished with exit code 0
