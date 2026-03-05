# Instruction Format Density Benchmark

**How compressed can an LLM system prompt get before the model stops following instructions?**

13 instruction formats. Same rules. 6 frontier models. 1,872 scored calls.

**[View the full results dashboard →](https://sapochat.github.io/instruct_density/)**

## Quick version

| What | Finding |
|---|---|
| **Sweet spot** | `compressed_nl` — 88% compliance at 41% of markdown size |
| **The cliff** | Below 200 characters, compliance drops to 58% |
| **First to break** | Concise responses and paragraph limits |
| **Last to break** | Factual accuracy (correct answers survive even extreme compression) |
| **Most resilient model** | Gemini 3.1 Pro Preview (-23pp from markdown to extreme) |
| **Most brittle model** | GLM-5 (-45pp from markdown to extreme) |

## Run it

```bash
# No dependencies beyond Python 3.10+ stdlib

# See all 13 generated system prompts
python instruction_density_bench.py --show-prompts --no-score

# Full benchmark (elite preset: Opus 4.6, Gemini 3.1 Pro, DeepSeek V3.2, GLM-5, Kimi K2.5, Qwen 3.5)
python instruction_density_bench.py --openrouter-key YOUR_KEY

# Quick smoke test
python instruction_density_bench.py --openrouter-key YOUR_KEY \
  --models anthropic/claude-opus-4.6 z-ai/glm-5 \
  --formats markdown hex_tagged \
  --tests identity guardrail_medical --no-score

# Just the inhuman formats
python instruction_density_bench.py --openrouter-key YOUR_KEY --inhuman-only
```

Presets: `elite` (default), `frontier`, `mid`, `cheap`, `smol`, `all`

## What's in the box

- `instruction_density_bench.py` — the benchmark script
- `bench_results.json` — results from the March 2026 run
- `index.html` — results dashboard (host via GitHub Pages)

## License

MIT
