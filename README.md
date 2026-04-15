# Omate POC — Real-time Clinical AI

> Companion code for the Medium article: *"A Scammer Sent Me the Best System Design Problem I've Seen This Year"*

A runnable proof-of-concept of the Omate healthcare AI system: real-time ECG denoising, zero-hallucination clinical RAG, and an agentic LangGraph monitoring pipeline.

**No GPU. No API key. No cloud account needed** — everything runs locally with synthetic ECG signals and a mock LLM.

---

## Quickstart (3 commands)

```bash
git clone https://github.com/yourname/omate-poc
cd omate-poc
make install && make demo-full
```

Or without `make`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy PyWavelets rich python-dotenv pytest
pip install -e .

python -m omate.demo_signal   # signal pipeline only
python -m omate.demo_full     # full system
pytest tests/ -v              # 56 tests
```

---

## What It Demonstrates

```
Synthetic ECG  ──►  Bandpass + Wavelet denoise  ──►  PatchTST anomaly detection
                                                              │
                                                              ▼
                                                     FHIR Mock Store
                                                     (Patient, Observation, Flag)
                                                              │
                                                              ▼
                                              Clinical RAG Engine
                                              Citation grounding +
                                              SelfCheckGPT consistency +
                                              Hard refusal < 0.80 confidence
                                                              │
                                                              ▼
                                              LangGraph Agent Graph
                                              Supervisor → Signal Analyst
                                              → RAG Reporter → HITL Review
                                              → Escalation (high-risk cases)
```

---

## Project Structure

```
omate-poc/
├── omate/
│   ├── signal/
│   │   ├── denoising.py      # bandpass + db4 wavelet + Pan-Tompkins
│   │   ├── anomaly.py        # PatchTST Transformer (CPU, ~500K params)
│   │   └── pipeline.py       # end-to-end → SignalEvent dataclass
│   ├── fhir/
│   │   └── store.py          # FHIR R4 mock: Patient, Observation, Flag,
│   │                         # DiagnosticReport, MedicationStatement
│   ├── rag/
│   │   ├── llm.py            # MockLLM + OpenAI + Ollama adapters
│   │   ├── guards.py         # citation grounding + SelfCheckGPT
│   │   └── engine.py         # full RAG pipeline
│   ├── agent/
│   │   ├── state.py          # OmateState TypedDict
│   │   ├── tools.py          # FHIR query tools
│   │   └── graph.py          # LangGraph graph + state machine fallback
│   ├── demo_signal.py
│   └── demo_full.py
├── tests/                    # 56 tests (pytest)
├── scripts/
│   └── download_mitbih.py    # MIT-BIH real ECG data
├── Makefile
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## Configuration

```bash
cp .env.example .env
```

```env
# Default: mock LLM (no API key needed)
LLM_BACKEND=mock

# Use OpenAI instead:
# LLM_BACKEND=openai
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini

# Use local Ollama (must be running: ollama serve + ollama pull mistral):
# LLM_BACKEND=ollama
# OLLAMA_MODEL=mistral

# Thresholds
CONFIDENCE_THRESHOLD=0.80
ESCALATION_THRESHOLD=0.90
```

---

## Switching to Real Components

| Component | POC default | Production swap |
|:---|:---|:---|
| ECG data | Synthetic | Real wearable via AWS Kinesis |
| FHIR store | In-memory mock | AWS HealthLake |
| Vector DB | In-memory cosine | Pinecone |
| LLM | Mock / Ollama | BioMistral-7B via vLLM |
| Anomaly model | PatchTST (random weights) | Pre-trained + ONNX → Triton |
| Agent memory | In-memory dict | LangGraph + Redis checkpointer |

---

## Real ECG Data (MIT-BIH)

Download ~50 MB of real cardiologist-annotated ECG recordings from PhysioNet:

```bash
# 1. Install wfdb reader
pip install wfdb

# 2. Download 11 records (Normal, AFib, V-tach, LBBB, PVC, Pacemaker, ...)
python scripts/download_mitbih.py

# 3. Run signal demo with real ECG
python -m omate.demo_signal --data mitbih --record 202   # AFib (best demo)
python -m omate.demo_signal --data mitbih --record 100   # Normal sinus rhythm
python -m omate.demo_signal --data mitbih --record 207   # Ventricular tachycardia
python -m omate.demo_signal --data mitbih                # All downloaded records
python -m omate.demo_signal --data mitbih --list         # List available records

# 4. Full pipeline with real ECG
python -m omate.demo_full --data mitbih --record 202
```

Each demo shows ground-truth cardiologist annotations vs model predictions.

---

## All Commands (without make)

```bash
# Install
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy PyWavelets rich python-dotenv pytest
pip install -e .

# Demos — synthetic ECG (no download needed)
python -m omate.demo_signal          # signal pipeline
python -m omate.demo_full            # full system: signal + FHIR + RAG + agent
python -m omate.demo_dashboard       # live terminal dashboard

# Demos — real MIT-BIH ECG
pip install wfdb
python scripts/download_mitbih.py
python -m omate.demo_signal --data mitbih --record 202
python -m omate.demo_full   --data mitbih --record 202

# Tests
pytest tests/ -v

# Real LLM backends
LLM_BACKEND=openai  OPENAI_API_KEY=sk-...     python -m omate.demo_full
LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... python -m omate.demo_full
LLM_BACKEND=ollama  python -m omate.demo_full   # requires: ollama serve
```

---

## References

1. Nie et al. (2023) *PatchTST* — ICLR 2023
2. Labrak et al. (2024) *BioMistral* — ACL 2024 Findings
3. Manakul et al. (2023) *SelfCheckGPT* — EMNLP 2023
4. Pan & Tompkins (1985) *QRS Detection* — IEEE Trans Biomed Eng
5. Donoho & Johnstone (1994) *Wavelet shrinkage* — Biometrika

---

## License

MIT
