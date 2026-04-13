"""
Clinical RAG Engine.

Retrieves relevant context from ChromaDB vector store,
generates a clinical report, then runs hallucination guards.

Production swap:
  - ChromaDB → Pinecone
  - MockLLM → BioMistral-7B via vLLM
  - simple_nli_consistency → SelfCheckNLI (mDeBERTa)
"""

import json
import textwrap
from dataclasses import dataclass

from .llm import LLMBackend, get_llm_backend
from .guards import run_hallucination_guards, GuardResult


CLINICAL_REPORT_PROMPT = """\
You are generating a clinical summary for physician review.

STRICT RULES:
1. Every factual claim MUST end with [SOURCE: resource_id=XXX]
2. If you cannot cite a specific record, DO NOT make the claim
3. Use ONLY data from the provided context below. Infer nothing.
4. If confidence is low, say so explicitly.

PATIENT CONTEXT:
{context}

SIGNAL EVENTS:
{signal_events}

Generate a concise clinical summary (3-5 sentences). Every sentence with a
factual claim must have a [SOURCE: resource_id=XXX] citation.
"""


@dataclass
class RAGResult:
    status: str                  # "READY_FOR_REVIEW" | "INSUFFICIENT_CONFIDENCE" | "ERROR"
    report: str | None
    confidence: float
    guard_result: GuardResult | None
    context_used: list[str]      # resource IDs retrieved
    message: str                 # human-readable status


class ClinicalRAGEngine:
    """
    Clinical RAG engine with hallucination guards.

    Args:
        fhir_store: FHIRStore instance for citation verification
        llm: LLM backend (defaults to env-configured backend)
        vector_store: optional ChromaDB collection for retrieval
        confidence_threshold: minimum confidence to emit a report
    """

    def __init__(self, fhir_store, llm: LLMBackend | None = None,
                  confidence_threshold: float = 0.80):
        self.fhir_store = fhir_store
        self.llm = llm or get_llm_backend()
        self.confidence_threshold = confidence_threshold
        self._chroma = None
        self._collection = None
        self._setup_vector_store()

    def _setup_vector_store(self):
        """Initialize simple in-memory vector store with clinical knowledge snippets."""
        self._kb_docs = []
        self._seed_knowledge_base()

    def _simple_embed(self, text: str) -> list[float]:
        """Bag-of-words embedding — no model download needed."""
        import hashlib
        words = set(text.lower().split())
        vec = [0.0] * 64
        for w in words:
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % 64
            vec[idx] += 1.0
        norm = sum(v**2 for v in vec) ** 0.5
        return [v / norm if norm > 0 else 0.0 for v in vec]

    def _seed_knowledge_base(self):
        """Add basic clinical reference snippets to in-memory KB."""
        docs = [
            "Atrial fibrillation (AFib) is characterized by irregular R-R intervals and absent P waves on ECG.",
            "ST elevation of >1mm in two contiguous leads may indicate acute myocardial infarction (STEMI).",
            "QT prolongation (QTc >450ms in men, >470ms in women) increases risk of torsades de pointes.",
            "Normal resting heart rate is 60-100 BPM. Bradycardia (<60) and tachycardia (>100) require evaluation.",
            "Left Bundle Branch Block (LBBB) presents with wide QRS (>120ms) and characteristic morphology.",
            "Amiodarone is a Class III antiarrhythmic known to prolong the QT interval.",
            "Fasting blood glucose >126 mg/dL on two occasions indicates diabetes mellitus.",
            "Hypoglycemia (<70 mg/dL) may cause palpitations, diaphoresis, and altered consciousness.",
        ]
        self._kb_docs = [
            {"id": f"kb-{i}", "text": d, "vec": self._simple_embed(d)}
            for i, d in enumerate(docs)
        ]

    def retrieve_context(self, query: str, n_results: int = 3) -> list[str]:
        """Retrieve relevant clinical knowledge snippets via cosine similarity."""
        if not self._kb_docs:
            return []
        q_vec = self._simple_embed(query)
        def cosine(a, b):
            dot = sum(x*y for x,y in zip(a,b))
            na = sum(x**2 for x in a)**0.5
            nb = sum(x**2 for x in b)**0.5
            return dot/(na*nb) if na*nb > 0 else 0.0
        scored = sorted(self._kb_docs, key=lambda d: cosine(q_vec, d["vec"]), reverse=True)
        return [d["text"] for d in scored[:n_results]]

    def generate_report(self, patient_id: str,
                         signal_events: list[dict]) -> RAGResult:
        """
        Generate a clinical report for a patient.

        Args:
            patient_id: FHIR Patient resource ID
            signal_events: list of signal event dicts from signal pipeline

        Returns:
            RAGResult with report text, confidence, and guard results
        """
        # Fetch patient context from FHIR store
        context = self.fhir_store.get_patient_context(patient_id)
        if not context.get("patient"):
            return RAGResult(
                status="ERROR",
                report=None,
                confidence=0.0,
                guard_result=None,
                context_used=[],
                message=f"Patient {patient_id} not found in FHIR store.",
            )

        # Build context string
        context_str = self._format_context(context)

        # Retrieve additional clinical knowledge
        query = " ".join(
            e.get("anomaly_class", "") for e in signal_events if e.get("is_anomaly")
        ) or "cardiac monitoring"
        knowledge_snippets = self.retrieve_context(query)

        if knowledge_snippets:
            context_str += "\n\nCLINICAL REFERENCE:\n" + "\n".join(
                f"- {s}" for s in knowledge_snippets
            )

        signal_str = json.dumps(signal_events, indent=2, default=str)

        # Build prompt
        prompt = CLINICAL_REPORT_PROMPT.format(
            context=context_str,
            signal_events=signal_str,
        )

        # Generate primary report
        primary_response = self.llm.generate(prompt, temperature=0.0)
        primary_report = primary_response.text

        # Get list of cited resource IDs from context
        context_resource_ids = self._extract_context_ids(context)

        def resource_exists(rid: str) -> bool:
            # Accept real FHIR store IDs + known context IDs + KB IDs
            return (
                self.fhir_store.resource_exists(rid)
                or rid in context_resource_ids
                or rid.startswith("kb-")
                or rid.startswith("obs-demo")   # demo data
            )

        def generate_variation(temperature: float) -> str:
            return self.llm.generate(prompt, temperature=temperature).text

        # Run hallucination guards
        guard = run_hallucination_guards(
            primary_report=primary_report,
            generate_fn=generate_variation,
            resource_exists_fn=resource_exists,
            n_samples=3,
            confidence_threshold=self.confidence_threshold,
        )

        if not guard.passed:
            return RAGResult(
                status="INSUFFICIENT_CONFIDENCE",
                report=None,
                confidence=guard.confidence,
                guard_result=guard,
                context_used=context_resource_ids,
                message=(
                    f"Insufficient confidence ({guard.confidence:.2f}) for automated "
                    "summary. Requires direct physician review. " + guard.reason
                ),
            )

        return RAGResult(
            status="READY_FOR_REVIEW",
            report=primary_report,
            confidence=guard.confidence,
            guard_result=guard,
            context_used=context_resource_ids,
            message=f"Report ready for physician review. Confidence: {guard.confidence:.2f}.",
        )

    def _format_context(self, context: dict) -> str:
        """Format FHIR context as a readable string for the LLM."""
        lines = []
        if context.get("patient"):
            p = context["patient"]
            name = p.get("name", [{}])[0].get("text", "Unknown") if isinstance(p.get("name"), list) else "Unknown"
            lines.append(f"PATIENT: {name} (ID: {p.get('id', 'unknown')})")
            lines.append(f"  Birth date: {p.get('birthDate', 'unknown')}")
            lines.append(f"  Gender: {p.get('gender', 'unknown')}")

        if context.get("observations"):
            lines.append("\nOBSERVATIONS (most recent first):")
            for obs in reversed(context["observations"][-5:]):
                obs_id = obs.get("id", "unknown")
                code = obs.get("code", {}).get("coding", [{}])[0].get("display", "Observation")
                value = obs.get("valueString") or str(obs.get("valueQuantity", {}).get("value", ""))
                lines.append(f"  [{obs_id}] {code}: {value}")

        if context.get("medications"):
            lines.append("\nMEDICATIONS:")
            for med in context["medications"]:
                drug = med.get("medicationCodeableConcept", {}).get("text", "Unknown")
                lines.append(f"  - {drug} (status: {med.get('status', 'unknown')})")

        if context.get("flags"):
            lines.append("\nACTIVE FLAGS:")
            for flag in context["flags"]:
                code = flag.get("code", {}).get("text", "Unknown")
                lines.append(f"  ⚠ {code}")

        return "\n".join(lines)

    def _extract_context_ids(self, context: dict) -> list[str]:
        """Extract all resource IDs from the patient context."""
        ids = []
        for obs in context.get("observations", []):
            if obs.get("id"):
                ids.append(obs["id"])
        for flag in context.get("flags", []):
            if flag.get("id"):
                ids.append(flag["id"])
        for med in context.get("medications", []):
            if med.get("id"):
                ids.append(med["id"])
        for report in context.get("recent_reports", []):
            if report.get("id"):
                ids.append(report["id"])
        return ids
