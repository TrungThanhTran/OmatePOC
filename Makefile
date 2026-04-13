.PHONY: install install-dev demo-signal demo-full test clean help

# ── Default ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Omate POC — Real-time Clinical AI"
	@echo ""
	@echo "  make install      Install core dependencies (CPU-only PyTorch)"
	@echo "  make demo-signal  Run signal intelligence demo"
	@echo "  make demo-full    Run full pipeline demo (signal + FHIR + RAG + agent)"
	@echo "  make test         Run all 56 tests"
	@echo "  make clean        Remove cache files"
	@echo ""

# ── Install ───────────────────────────────────────────────────────────────────
install:
	@echo "→ Installing core dependencies..."
	pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
	pip install numpy scipy PyWavelets rich python-dotenv pytest --quiet
	pip install -e . --quiet
	@echo "✓ Done. Run: make demo-signal"

install-dev: install
	@echo "→ Installing dev extras..."
	pip install pytest --quiet

# ── Run ───────────────────────────────────────────────────────────────────────
demo-signal:
	@cp -n .env.example .env 2>/dev/null || true
	python -m omate.demo_signal

demo-full:
	@cp -n .env.example .env 2>/dev/null || true
	python -m omate.demo_full

# ── Test ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

# ── With real LLM backends ────────────────────────────────────────────────────
demo-openai:
	@[ -n "$(OPENAI_API_KEY)" ] || (echo "Error: set OPENAI_API_KEY first" && exit 1)
	pip install openai --quiet
	LLM_BACKEND=openai python -m omate.demo_full

demo-ollama:
	@echo "→ Make sure Ollama is running: ollama serve"
	@echo "→ And you have a model: ollama pull mistral"
	LLM_BACKEND=ollama python -m omate.demo_full

# ── MIT-BIH real ECG data ─────────────────────────────────────────────────────
download-mitbih:
	pip install wfdb --quiet
	python scripts/download_mitbih.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache dist build *.egg-info
