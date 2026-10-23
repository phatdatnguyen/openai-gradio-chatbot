"""Bounded, source-scoped Compare and Evidence Check data validation.

Citation membership and verbatim quote occurrence are local checks. They do not
establish that a model's comparison or interpretation of a claim is correct.
"""

from .evidence import require_object, require_text, validate_source_ids


MAX_COMPARE_DOCUMENTS = 8
MAX_COMPARE_ASPECTS = 12
MAX_CLAIMS = 20
MAX_CLAIM_CHARACTERS = 1000
MAX_EVIDENCE_PER_CLAIM = 8
MAX_QUOTE_CHARACTERS = 1200
VERDICTS = ("supported", "contradicted", "mixed", "not_found")
STANCES = ("supports", "contradicts", "context")


def _text(value, *, limit):
    value = require_text(value, limit=limit)
    if any((ord(character) < 32 and character not in "\n\r\t")
           or ord(character) == 127 for character in value):
        raise ValueError("Review text cannot contain control characters.")
    return value


def _normalized(value):
    """Normalize whitespace only, preserving punctuation, case, and numbers."""
    return " ".join(value.split())


def parse_claims(claims_text):
    """Accept one bounded claim per nonempty line and assign app-owned IDs."""
    if not isinstance(claims_text, str) or len(claims_text) > MAX_CLAIMS * (MAX_CLAIM_CHARACTERS + 2):
        raise ValueError("Enter 1–20 claims, one per line, with at most 1,000 characters each.")
    claims = [{"id": f"c{index}", "text": line}
              for index, line in enumerate((line.strip() for line in claims_text.splitlines()
                                             if line.strip()), 1)]
    return validate_claims(claims)


def validate_claims(claims):
    """Validate the canonical claims saved alongside an Evidence Check result."""
    if not isinstance(claims, list) or not 1 <= len(claims) <= MAX_CLAIMS:
        raise ValueError("Enter 1–20 claims, one per line, with at most 1,000 characters each.")
    result = []
    for index, claim in enumerate(claims, 1):
        require_object(claim, {"id", "text"})
        if claim["id"] != f"c{index}":
            raise ValueError("Evidence Check claim identifiers must retain their original order.")
        text = _text(claim["text"], limit=MAX_CLAIM_CHARACTERS)
        if text != text.strip() or any(character in text for character in "\n\r"):
            raise ValueError("Enter each Evidence Check claim on its own line.")
        result.append({"id": claim["id"], "text": text})
    return result


def validate_compare_documents(documents):
    """Compare requires distinct selected documents before any paid work."""
    if not isinstance(documents, list) or not 2 <= len(documents) <= MAX_COMPARE_DOCUMENTS:
        raise ValueError("Compare requires between 2 and 8 selected documents.")
    identifiers = [document.get("id") if isinstance(document, dict) else None for document in documents]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Compare requires distinct selected document identifiers.")
    return identifiers


def _source_catalog(sources):
    if not isinstance(sources, list):
        raise ValueError("Review sources must be a source snapshot.")
    catalog = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str) or not source["id"] or source["id"] in catalog:
            raise ValueError("Review sources have invalid or duplicate identifiers.")
        catalog[source["id"]] = source
    return catalog


def validate_compare(content, *, documents, sources):
    """Validate all document cells and normalize their order without mutation."""
    require_object(content, {"aspects"})
    document_ids = validate_compare_documents(documents)
    catalog = _source_catalog(sources)
    aspects = content["aspects"]
    if not isinstance(aspects, list) or not 1 <= len(aspects) <= MAX_COMPARE_ASPECTS:
        raise ValueError("Compare requires between 1 and 12 comparison aspects.")
    seen_aspects, result = set(), []
    for aspect in aspects:
        require_object(aspect, {"aspect", "findings", "synthesis"})
        name = _text(aspect["aspect"], limit=160)
        key = _normalized(name).casefold()
        if key in seen_aspects:
            raise ValueError("Compare aspect names must be unique.")
        seen_aspects.add(key)
        findings = aspect["findings"]
        if not isinstance(findings, list) or len(findings) != len(document_ids):
            raise ValueError("Each comparison aspect must contain one finding for every selected document.")
        found, cited = {}, set()
        for finding in findings:
            require_object(finding, {"document_id", "status", "text", "source_ids"})
            identifier = finding["document_id"]
            if not isinstance(identifier, str) or identifier not in document_ids or identifier in found:
                raise ValueError("Compare returned an unknown or duplicate document finding.")
            status = finding["status"]
            if status not in ("found", "not_found"):
                raise ValueError("Compare findings must be found or not_found.")
            text = _text(finding["text"], limit=3000)
            ids = validate_source_ids(finding["source_ids"], catalog, allow_empty=status == "not_found")
            if status == "not_found" and ids:
                raise ValueError("A missing Compare finding must not claim supporting citations.")
            if any(catalog[source_id].get("document_id") != identifier for source_id in ids):
                raise ValueError("Each Compare finding must cite only its own document.")
            cited.update(ids)
            found[identifier] = {"document_id": identifier, "status": status, "text": text,
                                 "source_ids": list(ids)}
        if not cited:
            raise ValueError("Each comparison aspect needs at least one found, cited finding.")
        synthesis = aspect["synthesis"]
        require_object(synthesis, {"text", "source_ids"})
        synthesis_text = _text(synthesis["text"], limit=4000)
        synthesis_ids = validate_source_ids(synthesis["source_ids"], cited)
        result.append({"aspect": name, "findings": [found[identifier] for identifier in document_ids],
                       "synthesis": {"text": synthesis_text, "source_ids": list(synthesis_ids)}})
    return {"aspects": result}


def validate_evidence_check(content, *, claims, sources):
    """Check exact claims, quote occurrence, and the declared verdict's stances."""
    require_object(content, {"claims"})
    claims = validate_claims(claims)
    canonical = {claim["id"]: claim["text"] for claim in claims}
    catalog = _source_catalog(sources)
    rows = content["claims"]
    if not isinstance(rows, list) or len(rows) != len(claims):
        raise ValueError("Evidence Check must assess every requested claim exactly once.")
    found = {}
    for claim in rows:
        require_object(claim, {"id", "text", "verdict", "rationale", "evidence"})
        identifier = claim["id"]
        if not isinstance(identifier, str) or identifier not in canonical or identifier in found:
            raise ValueError("Evidence Check returned an unknown or duplicate claim identifier.")
        if claim["text"] != canonical[identifier]:
            raise ValueError("Evidence Check must preserve the original claim text.")
        verdict = claim["verdict"]
        if verdict not in VERDICTS:
            raise ValueError("Evidence Check returned an invalid verdict.")
        rationale = _text(claim["rationale"], limit=4000)
        evidence = claim["evidence"]
        if not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE_PER_CLAIM:
            raise ValueError("Each Evidence Check claim permits at most 8 evidence excerpts.")
        excerpts, seen, stances = [], set(), set()
        for item in evidence:
            require_object(item, {"source_id", "quote", "stance"})
            source_id = item["source_id"]
            if not isinstance(source_id, str) or source_id not in catalog:
                raise ValueError("Evidence Check returned an unknown source citation.")
            quote = _text(item["quote"], limit=MAX_QUOTE_CHARACTERS)
            text = catalog[source_id].get("text")
            if not isinstance(text, str) or _normalized(quote) not in _normalized(text):
                raise ValueError("Evidence Check quotes must occur verbatim in their cited original source.")
            stance = item["stance"]
            if stance not in STANCES:
                raise ValueError("Evidence Check returned an invalid evidence stance.")
            key = (source_id, _normalized(quote))
            if key in seen:
                raise ValueError("Evidence Check must not repeat the same cited excerpt for one claim.")
            seen.add(key)
            stances.add(stance)
            excerpts.append({"source_id": source_id, "quote": quote, "stance": stance})
        decisive = stances - {"context"}
        expected = {"supported": {"supports"}, "contradicted": {"contradicts"},
                    "mixed": {"supports", "contradicts"}, "not_found": set()}[verdict]
        if decisive != expected:
            raise ValueError("Evidence Check verdict must match the supporting and contradicting evidence stances.")
        found[identifier] = {"id": identifier, "text": canonical[identifier], "verdict": verdict,
                             "rationale": rationale, "evidence": excerpts}
    return {"claims": [found[claim["id"]] for claim in claims]}
