"""Check LFM comparison coverage, taxonomy isolation, and overlapping native masks."""
import copy
import json

import pytest

from scripts import compare_lfm as lfm


def native_span(kind, start, end):
    return {"type": kind, "start": start, "end": end}


def row(text="АБ ВГ", gold=None):
    return {"key": "organizer/one", "text": text, "dataset": "organizer", "split": "v1",
            "gold": {("PERSON", 0, 2)} if gold is None else gold, "uncertain": False, "traffic_weight": 1}


def decoders(*spans):
    return {"raw": list(spans), "hybrid": copy.deepcopy(list(spans))}


def test_native_all_types_union_keeps_unsupported_labels_and_overlaps_without_double_counting():
    rows = [row()]
    spans = decoders(native_span(lfm.PERSON_LABEL, 0, 2), native_span("credential.api_key", 0, 5),
                     native_span("healthcare.medication", 3, 5))
    output = lfm.native_masking(rows, {rows[0]["key"]: spans})
    for result in output["systems"].values():
        assert result["true_positive"] == 2
        assert result["false_positive"] == 2
        assert result["false_negative"] == 0
        assert result["precision"] == .5
        assert result["f1"] == .666667
        assert result["cases"] == 1


def test_native_full_masking_does_not_remove_unsupported_gold_categories():
    rows = [row(gold={("PERSON", 0, 2), ("UNSUPPORTED_PRIVATE_ID", 3, 5)})]
    native = {rows[0]["key"]: decoders(native_span(lfm.PERSON_LABEL, 0, 2))}
    result = lfm.native_masking(rows, native)["systems"]["raw"]
    assert (result["true_positive"], result["false_positive"], result["false_negative"]) == (2, 0, 2)
    assert result["recall"] == .5


def test_person_isolation_address_proxy_and_foreign_ids_remain_distinct_profiles():
    spans = [native_span(lfm.PERSON_LABEL, 0, 2), native_span(lfm.ADDRESS_LABEL, 3, 5),
             native_span("identity.ssn", 6, 8), native_span("identity.tax_id", 9, 11),
             native_span("identity.national_id", 12, 14)]
    before = copy.deepcopy(spans)
    person = lfm.native_candidates(spans)
    proxy = lfm.native_candidates(spans, address_proxy=True)
    assert person == [{"start": 0, "end": 2, "entity_type": "PERSON", "score": .85}]
    assert proxy == [*person, {"start": 3, "end": 5, "entity_type": "LOCATION", "score": .85}]
    assert spans == before


def test_overlong_gateway_profile_is_unavailable_without_partial_scoring_or_native_loss():
    first = row("А" * 201, {("PERSON", 0, 201)})
    second = {**row("БВ", {("PERSON", 0, 2)}), "key": "organizer/two"}
    rows = [first, second]
    native = {first["key"]: decoders(native_span(lfm.PERSON_LABEL, 0, 201)),
              second["key"]: decoders(native_span(lfm.PERSON_LABEL, 0, 2))}
    cache = {r["key"]: lfm.native_candidates(native[r["key"]]["raw"]) for r in rows}
    output, rejected = lfm.gateway_predictions(rows, {"lfm_raw_person_only": cache})
    assert set(output) == {"rules"}
    assert set(output["rules"]) == {first["key"], second["key"]}
    assert rejected == {"lfm_raw_person_only": [first["key"]]}
    assert lfm.native_masking(rows, native)["systems"]["raw"]["true_positive"] == 203


@pytest.mark.parametrize("invalid", [
    None, [], {}, {"raw": []}, {"raw": [], "hybrid": [], "other": []},
    {"raw": None, "hybrid": []}, {"raw": [None], "hybrid": []},
    {"raw": [{"start": True, "end": 2, "type": lfm.PERSON_LABEL}], "hybrid": []},
    {"raw": [{"start": 0, "end": 6, "type": lfm.PERSON_LABEL}], "hybrid": []},
    {"raw": [{"start": 0, "end": 2, "type": "invented"}], "hybrid": []},
    {"raw": [{"start": 0, "end": 2, "type": lfm.PERSON_LABEL, "score": .99}], "hybrid": []},
])
def test_native_validation_rejects_missing_decoders_bad_coordinates_unknown_types_and_invented_scores(invalid):
    with pytest.raises(ValueError):
        lfm.validate_native(invalid, "АБ ВГ", {lfm.PERSON_LABEL})


def write_cache(path, rows, native, protocol_hash="frozen-protocol"):
    path.write_text("".join(json.dumps({"case_id": r["key"], "text_sha256": lfm.text_digest(r["text"]),
                                       "decoders": native[r["key"]]}) + "\n" for r in rows))
    metadata = {"cache_sha256": lfm.golden.sha256(path), "protocol_sha256": protocol_hash}
    path.with_suffix(".meta.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize("change", ["text", "coverage", "order", "duplicate", "checksum", "protocol"])
def test_cache_cannot_be_reused_for_different_text_corpus_order_or_protocol(tmp_path, change):
    rows = [row(), {**row("ДЕ ЖЗ"), "key": "organizer/two"}]
    native = {r["key"]: decoders(native_span(lfm.PERSON_LABEL, 0, 2)) for r in rows}
    path = tmp_path / "lfm.jsonl"
    write_cache(path, rows, native)
    altered = copy.deepcopy(rows)
    if change == "text":
        altered[0]["text"] = "Другой текст"
    elif change == "coverage":
        altered.pop()
    elif change == "order":
        altered.reverse()
    elif change == "duplicate":
        write_cache(path, [rows[0], rows[0]], native)
    elif change == "checksum":
        path.write_text(path.read_text() + "\n")
    expected_protocol = "different-protocol" if change == "protocol" else "frozen-protocol"
    with pytest.raises(ValueError):
        lfm.load_lfm_cache(path, altered, {"native_types": [lfm.PERSON_LABEL]}, expected_protocol)


def test_cache_load_preserves_both_complete_native_decoders(tmp_path):
    rows = [row()]
    native = {rows[0]["key"]: decoders(native_span(lfm.PERSON_LABEL, 0, 2))}
    native[rows[0]["key"]]["hybrid"].append(native_span("healthcare.medication", 3, 5))
    path = tmp_path / "lfm.jsonl"
    write_cache(path, rows, native)
    result, _ = lfm.load_lfm_cache(path, rows, {"native_types": [lfm.PERSON_LABEL, "healthcare.medication"]},
                                  "frozen-protocol")
    assert result == native


def test_scoring_reports_full_native_false_positives_separately_from_person_isolation():
    rows = [row()]
    refs = {name: {rows[0]["key"]: [{"start": 0, "end": 2, "entity_type": "PERSON", "score": .85}]}
            for name in ("spacy", "gliner")}
    native = {rows[0]["key"]: decoders(native_span(lfm.PERSON_LABEL, 0, 2),
                                       native_span("org.company_name", 3, 5))}
    scored = lfm.score_corpus(rows, refs, native)
    assert scored["raw_model_person_isolation"]["systems"]["lfm_raw_person_only"]["typed_character"]["f1"] == 1
    assert scored["native_full_masking_all_types"]["systems"]["raw"]["false_positive"] == 2
    assert scored["native_prediction_category_counts"]["raw"] == {lfm.PERSON_LABEL: 1, "org.company_name": 1}
    assert scored["gateway_contract_rejections"] == {}
    assert "АБ ВГ" not in json.dumps(scored, ensure_ascii=False)
