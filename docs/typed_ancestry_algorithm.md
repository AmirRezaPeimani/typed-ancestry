# Typed-Ancestry Audit and Ancestor-Excluded Filler Selection

**Inputs.** Training records \(D_{\mathrm{tr}}\), evaluation records
\(D_{\mathrm{ev}}\), each tagged with dataset view and split; a set of
component-linking atom types
\(\mathcal{L}=\{\texttt{record},\texttt{context},\texttt{edit\_link}\}\);
target filler size \(m\); optional balance strata \(s(\cdot)\).

**Outputs.** Typed exposure tensor \(E\), connected-component identifiers
\(\kappa\), and an ancestor-excluded filler block \(F\).

```text
procedure CANONICALIZE(value)
    if value is text:
        return collapse_whitespace(NFKC(value))
    if value is a sequence:
        return [CANONICALIZE(x) for x in value]
    if value is a mapping:
        return {k: CANONICALIZE(value[k]) for k in sorted(keys(value))}
    return value

procedure ATOM(type, value, field, linkable)
    return (type, SHA256(canonical_JSON(CANONICALIZE(value))),
            field, linkable)

procedure EXTRACT_TYPED_ATOMS(view, record)
    A ← {ATOM(record, record, "__record__", true)}
    add a context atom for record.context or record.prompt, if present
    add reportable response atoms for every supported response field
    add reportable feedback atoms for every supported feedback field
    for each supported pair (original_response, edited_response):
        add ATOM(edit_link, {original, edited},
                 "original_response→edited_response", true)
    mark response and feedback atoms as exposure-reportable but not
        component-linking
    return unique sorted atoms A

procedure BUILD_COMPONENTS(records)
    index every atom by (atom.type, atom.hash)
    initialize one union–find node per record
    for each indexed atom key whose type is in L:
        union all records containing that atom
    assign each component the hash of its sorted member record identifiers
    return component identifiers κ

procedure COMPUTE_EXPOSURE(Dtr, Dev)
    index training atoms by (type, hash, source_view)
    for each evaluation record i, atom type t, and source view v:
        E[i,t,v] ← 1 iff any typed atom of i matches the training index
        also store matching-atom and matching-record counts
    return E

procedure SELECT_SAFE_FILLER(candidate_pool, Dev, m, strata)
    Cev ← {κ(r) : r ∈ Dev}
    eligible ← {r ∈ candidate_pool : κ(r) ∉ Cev}
    order eligible components by a seeded deterministic hash
    select whole components while matching m and the requested stratum counts
    reject and restart if any selected component intersects Cev
    assert |F| = m and {κ(r): r ∈ F} ∩ Cev = ∅
    return F

main
    canonicalize records and extract typed atoms
    κ ← BUILD_COMPONENTS(Dtr ∪ Dev ∪ candidate_pool)
    E ← COMPUTE_EXPOSURE(Dtr, Dev)
    F ← SELECT_SAFE_FILLER(candidate_pool, Dev, m, s)
    return E, κ, F
```

Exact ancestry and semantic similarity remain separate analyses. Canonical
field provenance is retained for inspection, while atom equality is determined
only by the pair `(atom type, canonical value hash)`.
