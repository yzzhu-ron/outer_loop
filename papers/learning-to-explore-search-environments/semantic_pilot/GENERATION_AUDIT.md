# Inspection of generated questions and query rewrites

The fixed examples reveal lost constraints, added premises, and questions whose
sampled source does not supply an answer. All 12 inspected probe-pair records and
12 inspected task-rewrite records passed format validation. That validation
therefore cannot establish semantic faithfulness or answerability.

This is an assistant-conducted inspection of fixed examples, not an independent
human review or a statistical estimate of semantic validity. Selection was fixed
before reading generated task examples or retrieval outcomes: the four records
in each corpus's `probe_audit_examples`, and the first four query IDs in its
`ids.test` list. Both orders are determined by the frozen preparation script.

Probe questions are checked against the full sampled source document when the
saved excerpt is insufficient. Task rewrites are checked against the original
query, without consulting qrels, retrieval ranks, relevance metrics, or action
selection outcomes. A source-supported question can still be broader or narrower
than its paired paraphrase. A schema-valid rewrite can change a scientific
claim's direction, presuppositions, or scope.

Hypothetical passages are synthetic retrieval representations. Factual-sounding
sentences in them are not verified answers. Their format warnings and the
generator's semantic-audit warnings remain in the saved records. The review
does not change prompts, exclusions, or evaluation data.

## SciFact

All eight inspected records passed format validation. Full source abstracts were
read for the four probe pairs. The findings below concern meaning and scope,
not retrieval performance.

| Sampled document | Question-pair inspection |
| --- | --- |
| `10538985` | Both questions ask how mouth/throat bacteria affect nitrite in exhaled samples. The indirect question spells out nitrate-to-nitrite conversion and healthy subjects. These details are supported by the source; the information need is well preserved. |
| `11578459` | The direct question asks about chromosome 7 gain, HOXA10 methylation, and the HOX signature. The indirect question assumes that the tumor “simultaneously silences the HOXA10 gene through hypermethylation.” That is a material simplification: the source specifically identifies an alternative HOXA10 promoter that **escapes** hypermethylation in HOX-high glioblastoma and discusses interactions between copy gain and methylation. The indirect wording adds a problematic premise. |
| `11935250` | The direct question concerns age-related methylation in mouse small intestine. The indirect version also asks whether the changes appear in lung, liver, and spleen. The source answers those comparisons, but this is a broader information need rather than a strict paraphrase. |
| `12086818` | Both concern lacritin's effect on secretion. The indirect question describes the protein rather than naming it, specifies unstimulated secretion, and adds ductal-cell growth. The source supports both effects; the second outcome broadens the direct question. |

| Test query | Rewrite inspection |
| --- | --- |
| `508`: “Hematopoietic Stem Cell purification reaches purity rate of up to 50%.” | The rewrite asks for methods achieving the stated 50% level, treating the claim as a premise. HyDE and subqueries introduce density-gradient centrifugation and magnetic beads, which the original query did not mention. These additions may alter retrieval toward particular methods; no evidence was consulted to validate them. |
| `805`: “Monoclonal antibody targeting of N-cadherin inhibits metastasis.” | The semantic rewrite asks about the effect on metastasis, weakening the original claim's explicit inhibitory direction but retaining its main entities. The second subquery asks how monoclonal antibodies inhibit metastasis without retaining **N-cadherin**, so it is less specific when searched independently. HyDE adds a synthetic mechanism. |
| `1180`: “The PRR MDA5 is a sensor of RNA virus infection.” | The semantic rewrite and both subqueries retain MDA5 and RNA-virus detection. The hypothetical passage adds mechanistic details about double-stranded RNA and interferon. Those additions were generated, not verified from evidence; no obvious information-need drift was found in the semantic rewrite. |
| `268`: “Cold exposure increases BAT recruitment.” | The semantic rewrite changes **recruitment** to **activation** of brown adipose tissue. These terms can describe different physiological outcomes; the wording risks substituting a related process for the requested one. The second subquery further specifies norepinephrine, which the original query did not mention. |

## FiQA

All eight inspected records passed format validation. One hypothetical passage
has a 39-word warning. The sampled source collection includes unanswered
questions and off-topic opinions as well as substantive finance discussion.

| Sampled document | Question-pair inspection |
| --- | --- |
| `108770` | The direct question asks whether liquidation value belongs in an operating company's accretion/dilution analysis. The indirect question adds IPO valuation, EPS, and the short-to-medium-term horizon. Those details are discussed in the source and largely preserve its information need. The source explicitly distinguishes IPO value from liquidation value; the generated “IPO valuation or liquidation value” phrasing compresses that distinction. |
| `139025` | Both questions ask why negative home equity matters. The supplied document is **itself an unanswered question** about that issue, not an explanation of the consequences. The pair is topical, but the generator's requirement that the document answer the question is unmet. Locating this known source would not establish that an answer was retrieved. |
| `146479` | The direct question asks what happens below a monthly average price of $1 on the NYSE. The indirect question asks how the NYSE determines delisting. The source is a comment claiming a threshold and describing the author's trades; it is not a detailed explanation of the exchange's decision process. It also asserts that delisting means losing everything, which this audit does not validate as a financial rule. Source matching and factual authority remain separate. |
| `152618` | The source briefly rejects comparing turkeys raised for food with child abuse. The generated questions ask for the basis of that comparison and ways the conditions resemble or differ. That requests more comparative analysis than the source supplies. This is also an off-topic opinion within the finance corpus, so its retrieval difficulty need not resemble finance-task difficulty. |

| Test query | Rewrite inspection |
| --- | --- |
| `5369`: “Paying for things on credit and immediately paying them off: any help for credit rating?” | The semantic rewrite substitutes paying the balance in full **each month** for paying purchases off **immediately**. It retains the credit-score topic but changes the timing condition the user asked about. The subqueries likewise emphasize timely payment and zero balance carryover, rather than the immediate-payment strategy. |
| `849`: “Accounting for reimbursements that exceed actual expenses” | The rewrite assumes employee reimbursements and an employer's accounting system. The original does not specify that setting. HyDE moves toward internal approval and audit procedures; these may be relevant in one interpretation but add context rather than merely clarifying it. |
| `7911`: “What is the difference between a 'trader' and a 'stockbroker'?” | The semantic rewrite and both subqueries preserve the comparison and the two roles. The hypothetical passage supplies simplified descriptions of the professions that were not fact-checked; its content remains synthetic. |
| `4678`: “Finance, Cash or Lease?” | The rewrite asks about “financing through cash or leasing,” collapsing the original **three-way choice**. Both subqueries compare cash with leasing and omit financing as a distinct option. HyDE additionally assumes a vehicle. Its 39-word format warning does not identify this semantic loss. |

## NFCorpus

All eight inspected records passed format validation. Two hypothetical passages
have word-budget warnings. Full abstracts were read for all four probe pairs.

| Sampled document | Question-pair inspection |
| --- | --- |
| `MED-1223` | The direct question asks about milk, growth, menarche, and a possible IGF-I role. The indirect question asks for biological mechanisms. The source reports associations, identifies IGF-I as a candidate, and explicitly says its mechanism is unknown. It supports discussion of the association and hypothesis, but does not supply the requested mechanistic explanation. |
| `MED-1707` | Both questions concern sugary drinks, weight gain, and metabolic disorders. The indirect version requests evidence, which the abstract supplies through meta-analyses and randomized trials. Its causal “contribute” wording is stronger than the direct question's “link,” but the central need and the reported evidence are substantially aligned. |
| `MED-1741` | Both questions concern Roundup's greater toxicity than glyphosate alone in placental cells. The indirect version adds disruption of aromatase and asks how adjuvants produce the effects. The source reports the comparative effects but only **suggests** increased bioavailability or bioaccumulation as an explanation. Treating that proposed mechanism as settled would go beyond the abstract. |
| `MED-1753` | Both questions ask how regulation, culture, and interest groups affect GM-animal commercialization in Europe and the United States. The short abstract says the article examines these topics but does not report the substantive findings. The pair matches the article's topic; the supplied text alone does not answer the questions. |

| Test query | Rewrite inspection |
| --- | --- |
| `PLAIN-1679`: “myelopathy” | The rewrite chooses an overview of causes, symptoms, and treatments. That is a plausible expansion of an underspecified term query, but the original supplies no reason to prefer those clinical aspects over other research questions. The 39-word HyDE passage is synthetic. |
| `PLAIN-2960`: “Pharmacists Versus Health Food Store Employees: Who Gives Better Advice?” | The semantic rewrite preserves the comparison. Subqueries focus on training and regulatory standards, which are possible explanations for advice quality rather than direct measurements of it. HyDE asserts that pharmacists generally give more clinically grounded recommendations; the original question does not establish that conclusion, and no evidence was checked here. |
| `PLAIN-441`: “Is apple cider vinegar good for you?” | The rewrite preserves a benefits-and-risks question while adding regular consumption. The first subquery narrows benefits specifically to blood-sugar control; the second emphasizes long-term side effects. HyDE includes a generic healthcare-consultation sentence, consuming retrieval text without adding query-specific evidence. |
| `PLAIN-2600`: “Eggs and Arterial Function” | The semantic rewrite broadens arterial **function** to arterial **health**. Subqueries select cholesterol and arterial stiffness. HyDE adds “Studies suggest” followed by a claim about moderate intake and elasticity, but no study was supplied or verified. Its 38-word warning addresses length rather than this unsupported evidence language. |

## What this inspection changes in the interpretation

These observations do not establish whether any action improves or harms
retrieval; the audit was completed without consulting retrieval outcomes. They
identify distinctions the experiment must preserve:

- A generated representation can retrieve useful material while dropping a
  requested option or changing a condition. Retrieval scores alone do not
  establish that an eventual answer would satisfy the original request.
- Recovering a known sampled document establishes a retrieval event. The source
  may be an unanswered question, an unsupported opinion, or an abstract that
  describes a topic without giving the requested explanation.
- Direct and indirect question pairs sometimes differ in scope or premises.
  Differences between these constructed families cannot be attributed solely to
  paraphrasing difficulty.

The sample is too small and was not drawn to estimate a population error rate.
A larger independent review of preserved constraints, source answerability,
and matched question-pair scope would test these concerns directly. No record
was removed and no frozen generation or evaluation code was changed following
this inspection.
