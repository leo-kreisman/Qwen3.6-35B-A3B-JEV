# Canonical form — the name of the problem behind the Courgette barrier

Asked whether "people passed through similar problems in another field." They did, in at least
seven, and the problem has a name. This is the survey, with sources, and then what it means
for us specifically.

## The problem, stated once

Two files can be **semantically identical while byte-different**, and the difference comes
entirely from things that carry no information about the content:

- **addressing** — the same thing referred to by an absolute location, which changes when
  anything moves
- **ordering** — the same set written in a different sequence
- **padding / encoding choices** — the same value written with or without redundant bytes
- **naming** — the same entity called different things

Courgette's barrier is one instance: an executable whose internal pointers are absolute, so a
one-line source change moves everything and every pointer's bytes change. The general shape is
older and wider than that.

## The property being restored has a name too

**Representation independence** — the value of an operation must not depend on which
representation of the same object was supplied. It appears under field-specific names:

- **alpha-equivalence** (type theory) — `λx.x` and `λy.y` are the same function; names don't matter
- **graph isomorphism** (graph theory) — two drawings of the same structure
- **canonical form / normalization** (algebra, rewriting) — one representative per equivalence class
- **canonical labelling** (combinatorics) — one numbering per isomorphism class
- **distinguished encoding** (cryptography) — one byte sequence per value
- **normalized representation** (genomics) — one spelling per variant

## The mechanism is the same in every field

Three steps, and every case below is an instance:

1. **Name the equivalence relation.** Say exactly which differences are not meaningful.
2. **Choose a canonical representative** deterministically — a total order, an index scheme, a
   minimal-length rule, a colour refinement.
3. **Compare or address canonically** — after step 2, equality is a byte or hash comparison,
   and "the same thing" is decidable cheaply.

**The key insight, and the reason this is one problem and not seven:** step 2 always works by
*removing a degree of freedom* and *replacing a concrete location with a symbolic index*. That
is Courgette. It is also DER, de Bruijn indices, InChI and canonical graph labelling.

---

## The seven fields, with what each actually does

### 1. Type theory — alpha-equivalence and de Bruijn indices

**Problem:** bound variable names carry no meaning, so `λx.x` ≡ `λy.y`, but they are different
bytes, and renaming can collide (capture).

**Mechanism:** **de Bruijn indices** (De Bruijn, 1972). Replace variable *names* with *indices
counting the number of binders between the use and its binding site*. Names disappear
entirely; alpha-equivalent terms become syntactically identical.

**This is Courgette's transformation with a different subject.** Name → index, where the index
is resolved relative to position instead of absolute. Courgette does it to addresses; De Bruijn
does it to variables.

- F. L. Bauer / N. G. de Bruijn, *Lambda calculus notation with nameless dummies* (1972)
- "A simple formalization of alpha-equivalence", arXiv:2507.10181 — α-equivalence classes and
  de Bruijn indices as the two standard routes
- *Proof Pearl: De Bruijn Terms Really Do Work*, NICTA — the isomorphism proof

**Transferable to us:** nothing directly. But it is the cleanest statement of the principle:
**if a datum is only a reference, store the reference structurally, not positionally.**

### 2. Chemistry — InChI and canonical SMILES

**Problem and how desperate:** the same molecule can be written in astronomically many ways.
Atom numbering is arbitrary, so the same structure yields different strings, and databases
cannot tell whether two records are the same compound. With millions of compounds and
decades of records, non-unique identifiers meant *silent duplication and wrong joins* — the
field's central data-quality crisis.

**Mechanism — InChI canonicalization, stated precisely:**

> "The InChI canonicalization algorithm **uniquely numbers the atoms of a molecule**."

Major step A, verbatim from the NIST discussion: the atoms in the "molecular graph" are given
numerical **"colors"** in order of precedence, then refined. Then hydrogens are stripped,
tautomers/charge/protonation are normalized, and the canonical numbering produces a single
string per structure.

Note what that is: **a colour refinement, then a canonical labelling** — the same algorithm as
nauty (below), independently invented in chemistry. And the same step as Courgette: the
arbitrary labels (atom numbers ↔ absolute addresses) are replaced by a canonical assignment.

- *InChI Canonicalization Algorithm*, depth-first.com — "uniquely numbers the atoms of a
  molecule. To date, the only implementation is found in the C source code of the InChI
  software" (i.e. the spec is *defined by its implementation* — a real-world failure mode, see
  "What this costs" below)
- IUPAC InChI; *The InChI Code*, J. Chem. Educ. 2018, doi:10.1021/acs.jchemed.8b00090

### 3. Genomics — VCF normalization (the closest analogue to our case)

**Problem and how desperate:** a variant (an indel) can be written in many equivalent ways,
and different variant callers choose different ones. From the Univ. of Michigan spec:

> "variant representation in VCF is **non-unique** for variants that have explicitly expressed
> reference and alternate sequences. **A failure to recognize this will frequently result in
> inaccurate analyses.**"

**Mechanism — two rules, both deterministic:**

- **Parsimony:** "representing a variant in as few nucleotides as possible without reducing the
  length of any allele to 0." *Right-trim and left-trim redundant bases.*
- **Left alignment:** "shifting the start position of that variant to the left till it is no
  longer possible to do so." *Shift the whole thing to the earliest legal position.*
- **Definition:** "A variant is **normalized** if and only if it is parsimonious **and** left
  aligned."

**Read that against Courgette.** Courgette: minimise the representation (symbolic form), then
canonically order what remains (sorted symbol table). Genomics: minimise the representation
(parsimony), then canonically position it (leftmost alignment possible). **Same two-axis
recipe — reduce redundancy, then impose a deterministic order.** Derived independently, in a
field with no contact with compiler toolchains.

Note also the *cost* they accepted, because it is our cost too: normalization is a
**preprocessing pass over every record**, it is **not idempotent-free** (you must normalize
both sides before comparing), and the spec documents cases **where the algorithm fails** —
so the canonical form is a *convention* maintained by tooling, not an intrinsic property of
the data. `bcftools norm`, `vt normalize`.

- *Unified representation of genetic variants*, Bioinformatics 2015, PMC4481842
- *Variant Normalization*, genome.sph.umich.edu — parsimony + left alignment, with proof
- VCF 4.5 spec, samtools.github.io/hts-specs/VCFv4.5.pdf

### 4. Graph theory — canonical labelling (nauty)

**Problem:** is graph A the same as graph B? Vertex labels are arbitrary, so byte comparison
says nothing, and there are n! labellings to check.

**Mechanism, and nauty's definition is the sharpest statement of the whole concept in any field:**

> "A canonical labelling map is a function C such that, for any graph G, partition π of V, and
> permutation γ of V, we have
> (a) C(G, π) = Gδ for some permutation δ such that πδ = c(π), and
> (b) **C(Gγ, πγ) = C(G, π)**.
> Informally, C relabels the vertices of G in order of colour, **ignoring the original vertex
> labels**."

**Clause (b) is the entire idea, formalised.** Apply any relabelling to the input; the output
is unchanged. That is exactly Courgette's `adjust` step stated as a mathematical law: renumber
the addresses however you like, the canonical form is identical — so the diff is empty where
the content is unchanged.

> **Theorem 1.** ... C(G1, π1) = C(G2, π2) **iff** G1 and G2 are isomorphic.

Which is to say: after canonicalization, isomorphism testing *becomes byte comparison*. That is
the payoff of every mechanism in this document.

- B. McKay & A. Piperno, *nauty/nug 2.4b3*, ANU — the definition above, Theorem 1
- *McKay's Canonical Graph Labeling Algorithm* — colour refinement + individualization and
  refinement; "exponential running time on some inputs, but performs exceptionally well under
  most circumstances"

**Direct reading for us:** nauty's colour refinement is a *cheap filter applied before the
expensive exact step*. That is precisely the role of a coarse layout key — e.g. a per-layer
stride signature that separates 589,824-row layers from 860,160-row layers **without touching
the data.** Which is how the blk.34 outlier was found here: a cheap structural check that
should have passed and didn't.

### 5. Cryptography — ASN.1 DER (distinguished encoding rules)

**Problem and how desperate:** signatures must verify. If the same value can be encoded several
ways, two parties compute different hashes over "the same" message — verification fails, or
worse, **an attacker obtains a second valid encoding of a signed message (signature
malleability).** Security-critical, so the field could not tolerate ambiguity.

**Mechanism, from the ECDSA/X.509 lineage:**

> "Integers are big-endian, two's-complement, and **minimal length**: leading zero bytes must be
> stripped. But the values are positive — so if the most significant bit of the first byte is
> set, a single `0x00` **must** be prepended."

Two rules, and they are the **same two rules as VCF**:
- **Minimality**: strip redundant leading zero bytes (= parsimony)
- **Deterministic disambiguation**: re-add exactly one `0x00` when the sign bit needs it (=
  left alignment, and also Courgette's canonical ordering — a rule that pins the last remaining
  freedom)

**And the field's own demonstration that this is not optional:** ECDSA has a second convention,
raw `r‖s` (IEEE P1363) — fixed-length, zero-padded, no ASN.1. The two coexist. Consequences
observed in practice:

- Signatures are **70–72 bytes in DER** (variable) vs **exactly 64 in raw** — so DER's
  minimality is a *compression* as well as a normalization
- WebCrypto emits raw only; OpenSSL emits DER only; **they cannot talk without conversion**
- **high-S / low-S**: two valid encodings of the same signature; Bitcoin adopted low-S as a
  relay rule (BIP-62, BIP-146) precisely to eliminate the second one
- Intermittent failures at "roughly 50%" are the signature of a non-normalized comparison

**This is the best cautionary example in the survey.** Two conventions in one field, both
"correct", and the mismatch causes real bugs. A canonical form is a *convention enforced
everywhere*, and partial adoption is worse than none — because it removes the error message
without removing the ambiguity.

- *ECDSA signature formats: DER vs raw (P1363), high-S and cross-library pitfalls*
- *ASN1*: Provably Correct, Non-malleable Parsing for ASN.1 DER*, ACM 10.1145/3573105.3575684
- RFC 8785 (JCS) — the JSON instance; abstract: "Cryptographic operations like hashing and
  signing need the data to be expressed in an **invariant format** so that the operations are
  reliably repeatable"

### 6. Build systems — reproducible builds

**Problem and how desperate:** you cannot verify that a distributed binary was built from the
claimed source if the build is not deterministic. Every rebuild differs because of **embedded
absolute paths, timestamps, and iteration order**. The supply-chain security guarantee was
*unavailable* until this was fixed — and it blocks binary transparency, SBOM integrity, and
independent rebuild verification.

**Mechanism, and note how blunt it is:**

> "Some tools will record the path of the source files in their output... In most cases however,
> **post-processing is required** to either remove the build path or to **normalize it to a
> predefined value**."

Concretely: `-fdebug-prefix-map=OLD=NEW` strips or rewrites directory prefixes from debug info;
`SOURCE_DATE_EPOCH` pins the timestamp; `BUILD_PATH_PREFIX_MAP` is a formal spec for
"information about the build-time filesystem layout, to generate reproducible output where all
embedded paths [agree]"; Rust/Cargo added `trim-paths`; `gzip -n` exists solely to avoid
writing the filename into the header.

**This is the most directly instructive case for us, for two reasons:**

1. **The failure mode is identical to ours.** An absolute path in debug info is a *pointer to a
   location that carries no content information* — exactly the absolute address in the
   executable that Courgette symbolifies, and exactly the problem of a tensor whose *offset*
   differs while its *content* is the same.
2. **The fix is a mapping, not a transformation.** `OLD=NEW` — you do not change the data, you
   declare an equivalence between two namespaces. That is the cheapest form of canonicalization
   in the survey, and it required **no change to the compiled content at all**.

- reproducible-builds.org/docs/build-path/
- BUILD_PATH_PREFIX_MAP specification
- Rust RFC 3127 `trim-paths`

### 7. Distributed filesystems — content-defined chunking (LBFS, the origin)

**Problem and how desperate:** synchronize large files over slow links. Fixed-size blocks fail
completely: from LBFS (SOSP 2001):

> "a single byte inserted at the start of a large file would **shift all the block boundaries,
> change the hashes of all the file's blocks, and thereby thwart any potential bandwidth
> savings**."

One byte, and the entire file re-transmits. This is *precisely* the "one line of source code
moves everything" pathology of Courgette, and it has the same cause: **structure was defined by
position instead of by content.**

**Mechanism — define the boundaries by content, not by position:**

> "it considers only non-overlapping chunks of files and avoids sensitivity to shifting file
> offsets by **setting chunk boundaries based on file contents, rather than on position within
> a file**. Insertions and deletions therefore only affect the surrounding chunks."

Implementation: examine every overlapping **48-byte region** and mark a boundary with
probability 2⁻¹³ (≈8 KiB average) when the **low 13 bits of a Rabin fingerprint** equal a chosen
value. 48 bytes was chosen empirically — "the effect of window size was not huge." This is
**content-defined chunking (CDC)**, and it is the origin of rsync's successors, zsync,
casync, zchunk, restic/borg, and HuggingFace Xet.

**Why this one is the deepest match to our problem:** LBFS's disease is not addresses, it is
**positional identity**. Courgette fixes pointers; LBFS fixes *boundaries*. Both ask: "what
part of this file's byte layout is carrying information, and what part is just where things
happen to sit?" Our expert pack is the same question asked of a GGUF — and LBFS is proof that
the answer can be obtained **without touching a single content byte**.

- Muthitacharoen, Chen, Mazières, *A Low-bandwidth Network File System*, SOSP 2001, §3.1.1
- *A Thorough Investigation of Content-Defined Chunking Algorithms for Data Deduplication*,
  arXiv:2409.06066

---

## Cross-field synthesis

| field | the ambiguity | canonical rule | what it buys |
|---|---|---|---|
| type theory | variable names | de Bruijn indices | alpha-equality = syntactic equality |
| chemistry | atom numbering | canonical atom labelling | one string per molecule |
| genomics | indel spelling | parsimony + left alignment | variant comparison is exact |
| graph theory | vertex labelling | canonical labelling `C(Gγ)=C(G)` | isomorphism = byte equality |
| cryptography | integer encoding | minimal + deterministic padding | signatures verify |
| build systems | embedded paths | `prefix-map OLD=NEW` | rebuilds are bit-identical |
| filesystems | positional boundaries | content-defined chunking | edits are local |

**Four properties hold in all seven:**

1. The rule is **deterministic** — no judgement calls, no heuristics at compare time.
2. It **removes a degree of freedom** rather than encoding around it.
3. Comparison afterwards is **cheap** — a byte compare or a hash.
4. The transformation is **not free**: it is an extra pass, it must be applied by *every*
   producer and consumer, and it is a *convention*, not a property of the data.

**And the recurring failure mode:** DER vs P1363 coexisting; InChI whose spec "is the C source
code"; the VCF spec documenting cases where its own algorithm fails. **A canonical form that is
only partly adopted makes things worse** — the ambiguity is still there, but it no longer
announces itself.

---

## What this means for us, concretely

**1. The general name for our barrier is: the format is already canonicalized.** This is the
positive version of the Courgette correction. The GGUF header is a **name → offset table**
(Courgette's symbol table), tensor order is fixed by a deterministic rule, byte layout is
32-byte aligned and declared. GGUF *is* a canonical container in exactly the sense of the table
above. So there is no un-symbolified addressing left, which is why Courgette had no operand.

**2. But one class of positional ambiguity remains, and it is the one we found.**
`data_start = 10,990,048` (≡ 480 mod 4096) means every expert tensor's *absolute* offset is
unaligned, while its *content* is fine. That is the **reproducible-builds** case exactly: a
location-dependent property (alignment) contaminating a content-identical artifact — and the
fix there was a **mapping**, not a transformation. Our pack does the same thing by
re-expressing layout, and it is why `O_DIRECT` went from 3840/3840 EINVAL to 1.00×
amplification with **zero content bytes changed**.

**3. Three ideas from this survey that we have not built, in order of fit:**

- **nauty's cheap-filter principle** — a coarse structural signature checked before the
  expensive path. Partially present: the per-layer stride class (1,769,472 vs 2,048,000 B) *is*
  such a signature, and it is what exposes blk.34/38/39. Not yet used as a fast path.
- **LBFS's content-defined boundaries** — for us: a *content-addressed* expert pack, where each
  slab is named by a hash of its contents rather than by `(layer, expert)`. Then a re-quantized
  checkpoint shares every unchanged slab with the previous one — **deltas between model
  versions, at zero cost to decode speed.** This is the closest thing in the survey to a
  genuine Courgette analogue for us, and it lives in the *transport/dedup* lane, not the decode
  lane. Worth noting honestly: it cannot speed up single-model inference.
- **DER/VCF minimality** — already satisfied by GGUF's quantization; there are no redundant
  bytes to strip, which the entropy measurement independently confirmed (6.96 of 8 bits/byte).

**4. What is definitively ruled out by this survey.** Every mechanism here is about
*representation*, and none of them changes the number of bytes that must cross the bus to
compute a token. The always-active 77.9% is not an addressing artifact and not an encoding
redundancy — it is 2.0143 GB of weights that must be read to produce one token, at 8.5 bpw.
**Canonicalization has nothing to say about it.** That remains a quantization question
(Q8_0 → ~Q4_5), and it remains where the 46% needed for 20 tok/s has to come from.

---

## Sources

Primary, read directly:

- nauty User's Guide 2.4b3 (`nug-2.4b3.pdf`) — canonical labelling definition, Theorem 1
- RFC 8785, JSON Canonicalization Scheme — invariant-format requirement
- *Variant Normalization*, U. Michigan — parsimony + left alignment, with proof
- Muthitacharoen et al., *A Low-bandwidth Network File System*, SOSP 2001 — §3.1.1, Rabin
  breakpoints, 48-byte window, 2⁻¹³ probability
- reproducible-builds.org, *Build path* — `-fdebug-prefix-map`, normalization to a predefined value
- *ECDSA signature formats: DER vs raw (P1363)* — minimality rule, high-S/low-S, the 50% failure

Secondary / named but not read in full:

- De Bruijn (1972), nameless dummies; arXiv:2507.10181 (alpha-equivalence formalization)
- InChI canonicalization (depth-first.com compilation of the NIST/InChI-discuss thread);
  J. Chem. Educ. 2018 doi:10.1021/acs.jchemed.8b00090
- *Unified representation of genetic variants*, Bioinformatics 2015, PMC4481842
- VCF v4.5 spec; *ASN1\*: Provably Correct, Non-malleable Parsing for ASN.1 DER*, ACM
  10.1145/3573105.3575684
- Rust RFC 3127 `trim-paths`; BUILD_PATH_PREFIX_MAP specification
- arXiv:2409.06066, content-defined chunking survey

Local copies of the primary sources: `~/.hermes/cache/scratch/canon/` (`nauty.txt`, `lbfs.txt`,
`rfc8785.txt`, `buildpath.txt`).
