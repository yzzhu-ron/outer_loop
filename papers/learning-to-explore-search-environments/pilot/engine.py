"""Fixed retrieval and query operations for the first, deliberately lexical pilot.

These actions are query-only heuristics, not semantic rewrites. ``body_terms`` is
an extractive, title-excluded diagnostic, not an indirect-query generator. Its
known-document rediscovery score does not establish relevance to real queries.
Neither transformations, query features nor probes consume task labels or corpus
statistics. Only the retrievers may inspect the full corpus.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

import numpy as np


ACTIONS = ("original", "keywords", "identifier_terms", "lead_clause")
PROBE_FAMILIES = ("exact_title", "body_terms")
FEATURE_NAMES = (
    "log_token_count", "log_character_count", "unique_token_fraction",
    "content_token_fraction", "identifier_fraction", "digit_token_fraction",
    "question_mark", "uppercase_token_fraction",
)
# One token retains common technical forms, e.g. IL-6, v1.2, C++, foo_bar, and
# /slash-separated/paths. Initial punctuation is discarded by tokenization.
_TOKEN = re.compile(r"\w+(?:[.:/#-]\w+)*(?:\+\+|#)?", re.UNICODE)
_STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does doing for from had
has have having how i if in into is it its may might more most much must my no
not of on or our ought shall she should so some such than that the their them
then there these they this those through to too under up us was we were what
when where which while who whom why will with would you your
""".split())


def _surface_tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def tokenize(text: str) -> list[str]:
    """Return Unicode word tokens, lowercasing but retaining internal identifiers."""
    return [token.lower() for token in _surface_tokens(text)]


def _is_identifier(token: str) -> bool:
    """A fixed lexical heuristic: numbers, separators, acronyms or camel case."""
    return (
        any(character.isdigit() for character in token)
        or any(character in "_.:/+#-" for character in token)
        or (len(token) >= 2 and token.isupper() and token.lower() not in _STOPWORDS)
        or re.search(r"[a-z][A-Z]", token) is not None
    )


def _unique_surface(tokens: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output = []
    for token in tokens:
        if token.lower() not in seen:
            seen.add(token.lower())
            output.append(token)
    return output


def transform(query: str, action: str, idf: Mapping[str, float] | None = None) -> str:
    """Apply a fixed query-only action, retaining every recognized identifier.

    ``idf`` is ignored for compatibility with callers passing backend statistics;
    this pilot's bounded target-access contract does not expose those statistics
    to the action library. ``identifier_terms`` includes all identifiers plus the
    first six distinct content terms. ``lead_clause`` retains up to 18 tokens
    from the first clause plus identifiers found elsewhere in the query.
    """
    if action not in ACTIONS:
        raise ValueError(f"Unknown action {action!r}; expected one of {ACTIONS}")
    if action == "original":
        return query.strip()
    tokens = _surface_tokens(query)
    if not tokens:
        return ""
    content = _unique_surface([
        token for token in tokens
        if token.lower() not in _STOPWORDS or _is_identifier(token)
    ])
    if action == "keywords":
        selected = content
    elif action == "identifier_terms":
        allowed = {token.lower() for token in content[:6]}
        selected = _unique_surface([
            token for token in tokens
            if token.lower() in allowed or _is_identifier(token)
        ])
    else:
        clause = re.split(r"(?:[;?!,]|\.(?=\s|$))", query, maxsplit=1)[0]
        selected = _unique_surface(
            _surface_tokens(clause)[:18]
            + [token for token in tokens if _is_identifier(token)]
        )
    # Stopword-only text remains a valid lexical query rather than disappearing.
    return " ".join(selected) if selected else query.strip()


def query_features(text: str) -> list[float]:
    """Eight fixed deployable features; no labels, retrieval or corpus access."""
    surface = _surface_tokens(text)
    lowered = [token.lower() for token in surface]
    count = len(surface)
    denominator = max(count, 1)
    return [
        math.log1p(count),
        math.log1p(len(text)),
        len(set(lowered)) / denominator,
        sum(token not in _STOPWORDS for token in lowered) / denominator,
        sum(_is_identifier(token) for token in surface) / denominator,
        sum(any(character.isdigit() for character in token) for token in surface) / denominator,
        float("?" in text),
        sum(len(token) >= 2 and token.isupper() for token in surface) / denominator,
    ]


def feature_bucket(text: str) -> int:
    """Four buckets: 2 * (more than 8 tokens) + any identifier-like token.

    The length threshold is below the 12-term body-probe cap, so that family can
    populate either length category without accessing real query distributions.
    """
    tokens = _surface_tokens(text)
    return 2 * int(len(tokens) > 8) + int(any(_is_identifier(token) for token in tokens))


def generate_probe(document: Mapping[str, object], family: str) -> str:
    """Generate a query using only one sampled document; empty means invalid.

    Exact title copying is the high-overlap control. Body terms exclude title
    words and stopwords, require at least three distinct terms, and take at most
    twelve terms spaced across the body. No selected terms occupy adjacent body
    token positions; a second check forbids shared three-token spans anywhere in
    the body, including repeated text. This is still extractive
    lexical overlap, not paraphrasing or a validated user-like query.
    """
    title = str(document.get("title") or "").strip()
    if family == "exact_title":
        return title if tokenize(title) else ""
    if family != "body_terms":
        raise ValueError(f"Unknown probe family {family!r}")
    excluded = set(tokenize(title)) | _STOPWORDS
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()
    previous_position = -2
    body_tokens = _surface_tokens(str(document.get("text") or ""))
    for position, token in enumerate(body_tokens):
        normalized = token.lower()
        if normalized in excluded or normalized in seen:
            continue
        if len(normalized) < 3 and not _is_identifier(token):
            continue
        if position - previous_position < 2:
            continue
        candidates.append((position, token))
        seen.add(normalized)
        previous_position = position
    if len(candidates) < 3:
        return ""
    count = min(12, len(candidates))
    indices = [round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)]
    body_lowered = [token.lower() for token in body_tokens]
    source_trigrams = {tuple(body_lowered[index:index + 3]) for index in range(len(body_lowered) - 2)}
    selected: list[str] = []
    for index in indices:
        token = candidates[index][1]
        if len(selected) >= 2 and tuple(item.lower() for item in selected[-2:] + [token]) in source_trigrams:
            continue
        selected.append(token)
    return " ".join(selected) if len(selected) >= 3 else ""


def _prepare_documents(documents: Sequence[Mapping[str, object]]) -> tuple[list[str], list[str]]:
    ids = [str(document["_id"]) for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Document IDs must be unique")
    texts = [
        f"{document.get('title') or ''}\n{document.get('text') or ''}".strip()
        for document in documents
    ]
    return ids, texts


def _top_indices(scores: np.ndarray, k: int, *, positive_only: bool) -> np.ndarray:
    """Order by descending score, then corpus insertion position, including ties."""
    if k <= 0 or scores.size == 0:
        return np.empty(0, dtype=np.int64)
    valid = np.isfinite(scores)
    if positive_only:
        valid &= scores > 0
    candidates = np.flatnonzero(valid)
    if len(candidates) > k:
        candidate_scores = scores[candidates]
        cutoff = np.partition(candidate_scores, len(candidate_scores) - k)[len(candidate_scores) - k]
        # Include *all* boundary ties before applying the deterministic tie rule.
        candidates = candidates[candidate_scores >= cutoff]
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order[:k]]


class BM25:
    """Full-corpus BM25 (k1=1.2, b=0.75), with title and body concatenated once.

    Query-term multiplicity contributes linearly, as in the standard unsaturated
    query-TF convention. Only documents with positive lexical match scores are
    returned; an empty or wholly out-of-vocabulary query returns no documents.
    """

    def __init__(self, documents: Sequence[Mapping[str, object]], k1: float = 1.2, b: float = 0.75):
        from scipy.sparse import csr_matrix

        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
        self.document_ids, texts = _prepare_documents(documents)
        self.k1, self.b = float(k1), float(b)
        counts = [Counter(tokenize(text)) for text in texts]
        self._vocabulary: dict[str, int] = {}
        for document_counts in counts:
            for term in document_counts:
                self._vocabulary.setdefault(term, len(self._vocabulary))
        document_frequency = Counter(term for document_counts in counts for term in document_counts)
        total = len(counts)
        self.idf = {
            term: math.log1p((total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        lengths = np.array([sum(document_counts.values()) for document_counts in counts], dtype=np.float64)
        average_length = float(lengths.mean()) if total else 0.0
        rows, columns, values = [], [], []
        for position, document_counts in enumerate(counts):
            length_ratio = lengths[position] / average_length if average_length else 0.0
            denominator_offset = self.k1 * (1 - self.b + self.b * length_ratio)
            for term, frequency in document_counts.items():
                rows.append(position)
                columns.append(self._vocabulary[term])
                values.append(self.idf[term] * frequency * (self.k1 + 1) / (frequency + denominator_offset))
        self._weights = csr_matrix(
            (np.asarray(values, dtype=np.float64), (rows, columns)),
            shape=(total, len(self._vocabulary)),
        ).tocsc()

    def search(self, query: str, k: int = 10) -> list[str]:
        terms = Counter(tokenize(query))
        matched = [(self._vocabulary[term], count) for term, count in terms.items() if term in self._vocabulary]
        if not matched or k <= 0:
            return []
        columns, frequencies = zip(*matched)
        scores = np.asarray(self._weights[:, list(columns)] @ np.asarray(frequencies, dtype=np.float64)).ravel()
        return [self.document_ids[index] for index in _top_indices(scores, k, positive_only=True)]

    def batch_search(self, queries: Sequence[str], k: int = 10) -> list[list[str]]:
        return [self.search(query, k=k) for query in queries]

    __call__ = search


class DenseRetriever:
    """Pinned frozen sentence-transformer, normalized float32 dot-product search.

    Documents concatenate title and body; the tokenizer truncates each encoded
    input at ``max_seq_length`` (default 256 including special tokens). Full raw
    corpus content and order, model revision, truncation, library versions and
    normalization are hashed into the local embedding cache identity. Truncation
    can discard late evidence and is an explicit limitation of this cheap pilot.
    """

    def __init__(
        self,
        documents: Sequence[Mapping[str, object]],
        cache_dir: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        revision: str | None = None,
        *,
        batch_size: int = 64,
        device: str = "cpu",
        num_threads: int = 4,
        max_seq_length: int = 256,
        show_progress_bar: bool = False,
    ):
        if revision is None or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            raise ValueError("Dense retrieval requires a full 40-character model commit revision")
        if batch_size < 1 or max_seq_length < 2 or num_threads < 1:
            raise ValueError("batch_size/num_threads must be positive; max_seq_length must be >= 2")
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(num_threads)
        self.document_ids, texts = _prepare_documents(documents)
        self.batch_size = batch_size
        self.model_name, self.revision = model_name, revision
        self.max_seq_length = max_seq_length
        self.model = SentenceTransformer(model_name, revision=revision, device=device, trust_remote_code=False)
        model_limit = getattr(self.model.tokenizer, "model_max_length", max_seq_length)
        encoder_limit = getattr(self.model[0].auto_model.config, "max_position_embeddings", max_seq_length)
        if max_seq_length > min(model_limit, encoder_limit):
            raise ValueError("Requested max_seq_length exceeds the model/tokenizer position limit")
        self.model.max_seq_length = max_seq_length
        config = {
            "format_version": 1,
            "model_name": model_name,
            "revision": revision,
            "max_seq_length": max_seq_length,
            "normalize_embeddings": True,
            "dtype": "float32",
            "document_format": "str(title or '') + newline + str(text or ''); strip",
            "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
            "transformers_version": importlib.metadata.version("transformers"),
        }
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8"))
        # Length-delimited JSON records cover every character and ID, not only
        # the model-visible prefixes. Corpus order determines ranking tie breaks.
        for document in documents:
            record = json.dumps(
                [str(document["_id"]), str(document.get("title") or ""), str(document.get("text") or "")],
                ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(record).to_bytes(8, "big"))
            digest.update(record)
        self.cache_key = digest.hexdigest()
        cache_directory = Path(cache_dir)
        cache_directory.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_directory / f"dense-{self.cache_key}.npy"
        self.embedding_cache_path = self.cache_path
        metadata_path = self.cache_path.with_suffix(".json")
        expected_shape = (len(texts), self.model.get_sentence_embedding_dimension())
        cached = None
        if self.cache_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("cache_key") == self.cache_key and metadata.get("config") == config:
                    candidate = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
                    if candidate.shape == expected_shape and candidate.dtype == np.float32 and np.isfinite(candidate).all():
                        cached = candidate
            except (ValueError, OSError, json.JSONDecodeError):
                cached = None
        if cached is None:
            embeddings = self._encode(texts, show_progress_bar=show_progress_bar) if texts else np.zeros(expected_shape, dtype=np.float32)
            if embeddings.shape != expected_shape or not np.isfinite(embeddings).all():
                raise ValueError("Model produced malformed document embeddings")
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=cache_directory, suffix=".npy", delete=False) as handle:
                    temporary_path = Path(handle.name)
                    np.save(handle, embeddings, allow_pickle=False)
                os.replace(temporary_path, self.cache_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            metadata_path.write_text(json.dumps({"cache_key": self.cache_key, "config": config, "shape": list(expected_shape)}, indent=2) + "\n")
            cached = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
        self.embeddings = cached

    def _encode(self, texts: Sequence[str], *, show_progress_bar: bool = False) -> np.ndarray:
        return np.asarray(self.model.encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=show_progress_bar,
            precision="float32",
        ), dtype=np.float32)

    def search(self, query: str, k: int = 10) -> list[str]:
        return self.batch_search([query], k=k)[0]

    def batch_search(self, queries: Sequence[str], k: int = 10) -> list[list[str]]:
        result: list[list[str]] = [[] for _ in queries]
        if k <= 0 or not self.document_ids:
            return result
        valid = [(index, query) for index, query in enumerate(queries) if tokenize(query)]
        # Bounded score matrices avoid allocating tasks x full-corpus at once.
        for start in range(0, len(valid), self.batch_size):
            chunk = valid[start:start + self.batch_size]
            query_embeddings = self._encode([query for _, query in chunk])
            scores = query_embeddings @ self.embeddings.T
            for (position, _), row in zip(chunk, scores):
                result[position] = [self.document_ids[index] for index in _top_indices(row, k, positive_only=False)]
        return result

    __call__ = search


Dense = DenseRetriever


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    """Graded nDCG with gain 2**relevance - 1; unjudged documents have zero gain."""
    if k <= 0:
        return 0.0
    relevant = {str(doc_id): float(value) for doc_id, value in qrels.items() if float(value) > 0}
    ideal = sum((2**value - 1) / math.log2(rank + 2) for rank, value in enumerate(sorted(relevant.values(), reverse=True)[:k]))
    if ideal == 0:
        return 0.0
    seen: set[str] = set()
    actual = 0.0
    for rank, doc_id in enumerate(ranked[:k]):
        doc_id = str(doc_id)
        if doc_id not in seen:
            actual += (2**relevant.get(doc_id, 0.0) - 1) / math.log2(rank + 2)
            seen.add(doc_id)
    return float(actual / ideal)


def reciprocal_rank(ranked: Sequence[str], target: str) -> float:
    """Reciprocal first rank of the known sampled document, zero when absent."""
    target = str(target)
    return next((1.0 / (rank + 1) for rank, doc_id in enumerate(ranked) if str(doc_id) == target), 0.0)


def recall_at_k(ranked: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    """Fraction of positive-judgment documents retrieved, without duplicate credit."""
    relevant = {str(doc_id) for doc_id, value in qrels.items() if float(value) > 0}
    if not relevant or k <= 0:
        return 0.0
    return len({str(doc_id) for doc_id in ranked[:k]} & relevant) / len(relevant)


recall = recall_at_k
