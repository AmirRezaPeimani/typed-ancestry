# Typed-Ancestry Schematic Field Verification

Date: 2026-07-30

Dataset: `nvidia/HelpSteer3`

Pinned revision: `f6d145777bcbde96137596340fab89793acd1031`

## Source-file verification

The pinned source files were downloaded read-only and matched the release
manifest:

- `edit/train.jsonl.gz`: SHA-256
  `c0d4208ddedd1bb66845fd29ff8be2549e8dde7dda720af944bd9308df688df2`;
- `edit_quality/validation.jsonl.gz`: SHA-256
  `6c89444f8be151b62a0e0d9693d470fbd21ccd226c656fa788556ab650df8184`.

All 13,740 training `edit` rows contain nonempty `context`,
`original_response`, `edited_response`, and `feedback` fields.

All 163 validation `edit_quality` rows contain nonempty `context`,
`original_response`, `good_edited_response`, `bad_edited_response`, and
`feedback` fields. The schematic’s candidate edit \(r_1\) denotes one edited
candidate. In the controlled ancestry intervention, \(r_1\) is specifically
the `good_edited_response` used to locate the training donor.

## Exact ancestor relation

The frozen data-construction code indexes training `edit` rows by

```text
hash(normalize(original_response), normalize(edited_response))
```

and admits a target `edit_quality` row only when

```text
hash(normalize(original_response), normalize(good_edited_response))
```

matches that index. Thus the displayed \(r_0 \rightarrow r_1\) directed edit
is the exact relation used to define the naïve ancestor. Every selected target
has one assigned donor under the frozen construction.

## Primary component policy

Complete-record, context, and directed-edit atoms are component-linking.
Standalone response and feedback atoms are exact exposure endpoints but have
`linkable=False`; feedback equality therefore does not merge ancestry
components under the primary policy.

An exact reconstruction of the 512 selected targets and 512 safe-filler
identifiers found zero shared component-linking atoms across the target and
safe-filler sets. One exact feedback atom was shared, providing a concrete
case where exposure is reported without merging the records into one
component.

## Evidence paths

- `scripts/prepare_factorial_data.py`
- `src/ancestry_audit/canonical.py`
- `src/ancestry_audit/adapter_api.py`
- `src/ancestry_audit/graph.py`
- `manifests/helpsteer3_sources.json`
- `results/dataset_design/summary.json`
