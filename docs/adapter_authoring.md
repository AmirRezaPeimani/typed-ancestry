# Adapter authoring

An adapter manifest is a JSON object with:

- `dataset`: stable dataset identifier;
- `revision`: immutable source revision;
- `sources`: one object per view and split.

Each source declares `view`, `split`, `path`, and a `mapping` object. Supported
mapping keys are:

- `context`: one field containing the conversation or prompt;
- `responses`: zero or more standalone response fields;
- `feedback`: zero or more string or list-of-string fields;
- `edits`: zero or more objects with `original` and `edited` field names;
- `domain`, `language`, and `label`: optional scalar metadata fields.

Complete rows, contexts, and directed edits are component-linking atoms.
Responses and feedback are reportable exact exposures but are not linking
atoms, because short generic text can otherwise create giant components.

Before reading exposure outcomes:

1. pin the source revision and record file checksums;
2. inspect representative rows from every source;
3. freeze the mapping;
4. write a known-link and known-nonlink fixture;
5. decide which atom types may merge components;
6. keep exact and semantic matching in separate outputs; and
7. record unavoidable provenance ambiguity.

Adding a field because it increases an observed exposure rate invalidates a
confirmatory audit. Such changes belong in a new, explicitly exploratory
manifest.
