.PHONY: install install-dev install-anthropic install-openai install-mlflow \
        demo-signal demo-full demo-dashboard \
        demo-openai demo-anthropic demo-ollama \
        demo-dashboard-openai demo-dashboard-anthropic \
        test clean help

# ── Default ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Omate POC — Real-time Clinical AI"
	@echo ""
	@echo "  ── Install ──────────────────────────────────────────────────────"
	@echo "  make install               Core deps (CPU PyTorch, no API keys)"
	@echo "  make install-anthropic     + Anthropic SDK"
	@echo "  make install-openai        + OpenAI SDK"
	@echo "  make install-mlflow        + MLflow experiment tracker"
	@echo ""
	@echo "  ── Demos ────────────────────────────────────────────────────────"
	@echo "  make demo-signal           Signal denoising only (~2s)"
	@echo "  make demo-full             Full pipeline: signal+FHIR+RAG+agent"
	@echo "  make demo-dashboard        Live dashboard (mock LLM — no key needed)"
	@echo "  make demo-dashboard-anthropic  Dashboard with ANTHROPIC_API_KEY"
	@echo "  make demo-dashboard-openai     Dashboard with OPENAI_API_KEY"
	@echo ""
	@echo "  ── Tests ────────────────────────────────────────────────────────"
	@echo "  make test                  Run all tests"
	@echo ""

# ── Install ───────────────────────────────────────────────────────────────────
install:
	@echo "→ Installing core dependencies (CPU-only PyTorch)..."
	pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
	pip install numpy scipy PyWavelets rich python-dotenv pytest --quiet
	pip install -e . --quiet
	@echo "✓ Done. Run: make demo-dashboard"

install-dev: install
	pip install pytest --quiet

install-anthropic: install
	@echo "→ Installing Anthropic SDK..."
	pip install anthropic --quiet
	@echo "✓ Set ANTHROPIC_API_KEY then: make demo-dashboard-anthropic"

install-openai: install
	@echo "→ Installing OpenAI SDK..."
	pip install openai --quiet
	@echo "✓ Set OPENAI_API_KEY then: make demo-dashboard-openai"

install-mlflow: install
	@echo "→ Installing MLflow..."
	pip install mlflow --quiet
	@echo "✓ Set MLFLOW_TRACKING_URI to connect to your MLflow server"

# ── Demos ─────────────────────────────────────────────────────────────────────
demo-signal:
	@cp -n .env.example .env 2>/dev/null || true
	python -m omate.demo_signal

demo-full:
	@cp -n .env.example .env 2>/dev/null || true
	python -m omate.demo_full

demo-dashboard:
	@cp -n .env.example .env 2>/dev/null || true
	python -m omate.demo_dashboard --no-interactive

demo-dashboard-anthropic:
	@[ -n "$(ANTHROPIC_API_KEY)" ] || (echo "Error: export ANTHROPIC_API_KEY=sk-ant-..." && exit 1)
	pip install anthropic --quiet
	ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) LLM_BACKEND=anthropic \
	  python -m omate.demo_dashboard --no-interactive

demo-dashboard-openai:
	@[ -n "$(OPENAI_API_KEY)" ] || (echo "Error: export OPENAI_API_KEY=sk-..." && exit 1)
	pip install openai --quiet
	OPENAI_API_KEY=$(OPENAI_API_KEY) LLM_BACKEND=openai \
	  python -m omate.demo_dashboard --no-interactive

# ── Legacy full-pipeline demos with real backends ─────────────────────────────
demo-openai:
	@[ -n "$(OPENAI_API_KEY)" ] || (echo "Error: export OPENAI_API_KEY=sk-..." && exit 1)
	pip install openai --quiet
	LLM_BACKEND=openai python -m omate.demo_full

demo-anthropic:
	@[ -n "$(ANTHROPIC_API_KEY)" ] || (echo "Error: export ANTHROPIC_API_KEY=sk-ant-..." && exit 1)
	pip install anthropic --quiet
	LLM_BACKEND=anthropic python -m omate.demo_full

demo-ollama:
	@echo "→ Ensure Ollama is running: ollama serve"
	@echo "→ And you have a model:     ollama pull mistral"
	LLM_BACKEND=ollama python -m omate.demo_full

# ── Test ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

# ── MIT-BIH real ECG data ─────────────────────────────────────────────────────
download-mitbih:
	pip install wfdb --quiet
	python scripts/download_mitbih.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache dist build *.egg-info
