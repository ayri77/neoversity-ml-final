# Final Project Delivery Checklist

> **Internal working document — do not link from the public README.**
>
> This file is the single source of truth for final-project completion. No task is
> complete without explicit evidence. Results, dates, identities, and external scores
> must be verified immediately before publication.

Allowed status values:

- `TODO`
- `IN PROGRESS`
- `BLOCKED`
- `READY FOR REVIEW`
- `DONE`
- `DEFERRED`

## 1. Deadline and current state

| Item | State at document creation |
| --- | --- |
| LMS deadline | **2026-08-02 23:45 Europe/Berlin** |
| Branch | `feature/prepared-dataset-pipeline-v1` |
| Commit | `d94967673d0278e294d175165bfb52065c1dc808` |
| Upstream | `origin/feature/prepared-dataset-pipeline-v1` |
| Best verified Kaggle Public Score | **0.9112** |
| Best-score evidence | `v3_targeted_missingness`, AutoGluon-screened frozen `LightGBMPrep_r31`, threshold 0.117 |
| Final academic candidate | Not frozen |
| English README | Draft created; review and screenshots pending |
| Final LMS notebook | Not created |
| Working tree | Dirty because documentation, notebook, and focused AutoGluon work are proceeding in parallel |

Known parallel state on 2026-08-01:

- the focused v3 AutoGluon smoke run
  `ag-v3-focused-hybrid-smoke-s42-20260801-173849` has a local `_SUCCESS`
  marker and reports `WeightedEnsemble_L2` as its best model;
- that smoke artifact still requires result and resource review before it informs a
  full experiment;
- focused v3 and v6 configuration files are present as untracked parallel work;
- AutoGluon profile code, tests, runner documentation, the operational handoff, and
  notebook 03 have concurrent working-tree changes;
- no completed focused full run was identified when this checklist was created.

**Warning:** experiment results and the final candidate may change. Recheck artifacts,
Git state, and external Kaggle evidence before replacing any dynamic value.

## 2. Completion workflow

Follow these phases in dependency order. Do not start an expensive optional task when
it would put the LMS submission at risk.

### Phase A — public English README

1. Draft the English README.
2. Review every public claim, command, result, and link.
3. Capture the approved Control Panel screenshots.
4. Review screenshots for personal or sensitive information.
5. Insert only reviewed screenshots at the existing README markers.
6. Re-run link, secret-pattern, formatting, and status checks.

### Phase B — final experiment decision

1. Review the completed focused smoke artifact.
2. Decide whether a focused full experiment is still justified by time and resources.
3. Run the authorized focused full experiment only under the separate AutoGluon
   workflow and only once its config is frozen.
4. Compare focused evidence with project-owned manual and Research v2 evidence.
5. Select the final candidate using protocol-aware evidence.
6. Decide whether blending is justified by compatible aligned OOF predictions.
7. Defer optional dataset/model experiments unless they can change the decision without
   endangering the deadline.

### Phase C — freeze and reproduce the final submission

1. Freeze the dataset ID and manifest identity.
2. Freeze the candidate/model identity and implementation category.
3. Freeze seed(s), evaluation protocol, threshold, and any blend weights.
4. Freeze the exact source run and source config.
5. Create a thin final orchestration notebook.
6. Run it from a clean kernel in the documented environment.
7. Verify that it trains or executes the selected fixed workflow.
8. Verify exactly 2,500 predictions and exact columns `index,y`.
9. Verify row order, binary labels, threshold, positive count, and CSV SHA-256.
10. Compare the generated CSV with the intended Kaggle file.
11. Submit the final Kaggle candidate if applicable and record external evidence.

### Phase D — Ukrainian academic documentation

1. Create `README.uk.md`.
2. Create or update `docs/final-project-report.uk.md`.
3. Add the frozen final results and AutoGluon disclosure to both READMEs.
4. Prepare the Ukrainian forum publication.
5. Cross-check the notebook, report, READMEs, benchmark ledger, and checklist.

### Phase E — delivery and publication

1. Upload the exact validated notebook once to LMS.
2. Add the repository link.
3. Re-open and verify the uploaded LMS artifact.
4. Record the upload timestamp and evidence.
5. Create the final Git checkpoint only after all intended files are reviewed.
6. Publish the forum post within the allowed period.
7. Save publication evidence and URL.
8. Move cleanup and platform generalization to the post-deadline backlog.

## 3. Master task table

| ID | Priority | Task | Owner | Status | Dependency | Expected artifact/evidence | Last verified date | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A01 | P0 | Draft complete English public README | Codex | READY FOR REVIEW | None | `README.md` diff | 2026-08-01 | Claims checked against code, manifests, CLIs, and local benchmark evidence |
| A02 | P0 | Review README wording and scope | Pavlo | TODO | A01 | Explicit approval or review notes | 2026-08-01 | Check academic tone and project ownership |
| A03 | P1 | Capture approved Control Panel screenshots | Pavlo | TODO | A02; stable UI state | Five reviewed PNG files | 2026-08-01 | Follow section 5 exactly |
| A04 | P1 | Insert screenshots into README | Codex | TODO | A03 | README image links render | 2026-08-01 | Do not insert unreviewed images |
| A05 | P0 | Final README validation | shared | TODO | A02; A04 or screenshot deferral | Link/format/secret check log | 2026-08-01 | Screenshots may be deferred if time is critical |
| B01 | P0 | Review focused v3 smoke result | shared | READY FOR REVIEW | Smoke `_SUCCESS` | Review note with leaderboard, duration, resource warnings, and decision | 2026-08-01 | Artifact completed; review is not complete |
| B02 | P0 | Freeze focused full config and decide whether to run | Pavlo | TODO | B01 | Signed-off config identity or explicit skip decision | 2026-08-01 | Check whether parallel v3/v6 config work supersedes current files |
| B03 | P0 | Execute focused full experiment | Assistant | BLOCKED | B02; explicit authorization; resources | Terminal run artifact and inspection summary | 2026-08-01 | Do not launch merely because config exists |
| B04 | P0 | Compare focused evidence with manual/Research v2 evidence | shared | BLOCKED | B01 and B03 or explicit B03 skip | Protocol-aware comparison note | 2026-08-01 | Do not compare unlike validation scores as equivalent |
| B05 | P0 | Select final dataset and candidate | Pavlo | BLOCKED | B04 | Decision-log entry plus frozen identities | 2026-08-01 | Must address AutoGluon/manual academic disclosure |
| B06 | P0 | Decide whether a blend is justified | shared | BLOCKED | B05; compatible OOF evidence | Fixed components/weights or explicit no-blend decision | 2026-08-01 | No leaderboard-tuned weights |
| B07 | P2 | Run optional extra dataset/model experiments | Assistant | DEFERRED | B05 indicates material uncertainty | Completed controlled artifacts | 2026-08-01 | Deadline protection takes precedence |
| C01 | P0 | Freeze dataset, candidate, seed, threshold, protocol, and submission identity | shared | BLOCKED | B05; B06 | Frozen decision record and source references | 2026-08-01 | One authoritative set of values |
| C02 | P0 | Create thin final orchestration notebook | Codex | BLOCKED | C01 | Tracked final `.ipynb` | 2026-08-01 | Import tested modules; avoid monolithic duplication |
| C03 | P0 | Review final notebook content | Pavlo | BLOCKED | C02 | Review approval | 2026-08-01 | Confirm it represents the submitted solution |
| C04 | P0 | Run final notebook from a clean kernel | Pavlo | BLOCKED | C03; environment/data ready | Successful Run All with fresh outputs | 2026-08-01 | Manual execution is the final LMS evidence |
| C05 | P0 | Validate final CSV schema, rows, order, labels, and identity | shared | BLOCKED | C04 | Validation report, SHA-256, positive count | 2026-08-01 | Exactly 2,500 rows and columns `index,y` |
| C06 | P0 | Compare final CSV with intended Kaggle file | shared | BLOCKED | C05 | Byte/hash comparison or documented difference | 2026-08-01 | Never claim reproduction when files differ |
| C07 | P0 | Submit final Kaggle candidate if applicable | Pavlo | BLOCKED | C06 | Kaggle submission record | 2026-08-01 | External manual action |
| D01 | P1 | Create Ukrainian academic README | Codex | TODO | C01 preferred | `README.uk.md` | 2026-08-01 | Keep public README English |
| D02 | P1 | Create/update Ukrainian final-project report | Codex | TODO | C01 preferred | `docs/final-project-report.uk.md` | 2026-08-01 | Detailed academic/forum material |
| D03 | P0 | Update final results across documents | shared | BLOCKED | C01; C05; C07 if used | Consistent values in update map | 2026-08-01 | Reverify every score/date |
| D04 | P1 | Prepare forum post | Pavlo | BLOCKED | D02; D03 | Final draft | 2026-08-01 | Include repository link and accurate disclosure |
| E01 | P0 | Upload exact notebook to LMS | Pavlo | BLOCKED | C04–C06; D03 as required | LMS receipt/screenshot | 2026-08-01 | Upload once after final verification |
| E02 | P0 | Add and verify repository link in LMS | Pavlo | BLOCKED | E01; public repository review | Visible working link | 2026-08-01 | Verify in submitted view |
| E03 | P0 | Record final Git checkpoint | Pavlo | BLOCKED | All intended delivery files reviewed | Final commit ID and clean/known status | 2026-08-01 | Do not include local data/artifacts |
| E04 | P1 | Publish forum post | Pavlo | BLOCKED | D04; LMS completed | Forum URL and timestamp | 2026-08-01 | Save evidence |
| E05 | P2 | Platform cleanup and generalization | shared | DEFERRED | Deadline passed | Backlog/roadmap artifacts | 2026-08-01 | Must not distract from delivery |

## 4. Manual actions for Pavlo

This section contains only actions that require or are best performed by Pavlo.

- [ ] Review and approve the English README.
- [ ] Capture the approved Control Panel screenshots listed in section 5.
- [ ] Verify that screenshots show no username, absolute personal path, token, secret,
      private dataset value, unintended run ID, or sensitive browser content.
- [ ] Record the current Kaggle leaderboard position with date, time, and timezone.
- [ ] Confirm the exact final Kaggle submission filename.
- [ ] Confirm the public score from Kaggle after submission processing.
- [ ] Record the exact submission CSV SHA-256.
- [ ] Record the predicted positive count and rate.
- [ ] Approve the final dataset, model/candidate, seed, threshold, and blend decision.
- [ ] Select and approve the exact final LMS notebook.
- [ ] Restart the kernel and perform **Run All** from top to bottom.
- [ ] Manually inspect notebook outputs and the generated CSV.
- [ ] Confirm exactly 2,500 data rows and exact columns `index,y`.
- [ ] Upload the exact validated notebook once to LMS.
- [ ] Add the public repository link to LMS.
- [ ] Re-open the submitted LMS entry and verify both file and link.
- [ ] Save a timestamped LMS receipt or screenshot.
- [ ] Publish the Ukrainian forum post.
- [ ] Save the final forum URL and timestamped publication evidence.

## 5. Screenshot capture plan

Do not capture a view until its selected artifact is validated and suitable for public
display. Crop browser chrome, local filesystem paths, usernames, hostnames, private
notes, and irrelevant job logs. Use a consistent light or dark theme.

| Proposed filename | Exact UI view | Required visible content | State/data to select | Hide or crop | Recommended size | README target | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `docs/images/control-panel-run.png` | Run → Train → Dataset-driven experiment | Dataset ID, target-dependency badge, model/template selectors, validation-ready state | A registered target-independent package and a smoke template; do not launch | Absolute paths, username, unrelated sidebar state, private config text | 1600×900 | Control Panel, after Run description | TODO |
| `docs/images/control-panel-results.png` | Results → Research Workspace | Filters, experiment inventory, Dataset × model matrix, comparability indicators | Completed validated Research v2 runs across multiple datasets/models | Notes containing local details, absolute artifact roots | 1600×900 | Control Panel, after Results description | TODO |
| `docs/images/control-panel-compare.png` | Run/Results → Compare → same-dataset paired comparison | Left/right candidates, compatibility state, key metric deltas | Two completed compatible runs on the same dataset and protocol | Full local run paths, irrelevant raw JSON | 1600×900 | Control Panel, paired comparison paragraph | TODO |
| `docs/images/control-panel-dataset-comparison.png` | Compare → Dataset Comparison | Parent/child dataset IDs, identity gates, descriptive deltas | A validated comparison such as LightGBM `v0_raw_minimal` vs `v1_missingness_summary` | Absolute artifact directory, unrelated failed comparisons | 1600×900 | Control Panel, cross-dataset paragraph | TODO |
| `docs/images/control-panel-submission.png` | Run → Generate submission | Candidate identity, readiness gates, authenticated asset state, explicit local-only action | Final approved candidate after identities are frozen | Local source paths, hashes not intended for public display, any sensitive confirmation data | 1600×900 | Control Panel, Generate submission paragraph | BLOCKED |

Do not capture the planned campaign-results UI or cross-dataset multi-blend UI; those
views are not complete.

## 6. Dynamic values register

`Current verified value` may show historical evidence, but that does not make it the
final value. Replace only after the final decision has evidence.

| Value | Current verified value | Authoritative source | Status | Update locations |
| --- | --- | --- | --- | --- |
| Final dataset ID | TBD; historical best uses `v3_targeted_missingness` (exploratory) | Frozen Dataset Package manifest and final decision log | BLOCKED | This checklist, README results/status, Ukrainian docs, final notebook, ledger |
| Final model/candidate ID | TBD; historical best is `LightGBMPrep_r31` discovered by AutoGluon | Frozen source run/config and decision log | BLOCKED | Same as above |
| Final blend components/weights | TBD; no final blend frozen | Compatible OOF blend artifact or explicit no-blend decision | BLOCKED | Checklist, READMEs, report, notebook |
| Seed(s) | TBD; focused smoke used seed 42 only | Final config/resolved config | BLOCKED | Checklist, report, notebook |
| Threshold | TBD; 0.117 belongs to the historical 0.9112 submission | Final threshold evidence | BLOCKED | Checklist, READMEs, ledger, report, notebook |
| Internal evaluation metric/protocol | Balanced Accuracy; final protocol TBD | Final validated run and threshold summary | BLOCKED | Checklist, READMEs, report, notebook |
| Submission filename | TBD; historical reference `autogluon_v3_extreme_threshold_0117.csv` | Generated final artifact plus Kaggle record | BLOCKED | Checklist, ledger, report, notebook |
| CSV SHA-256 | TBD; historical 0.9112 file: `620181c4a5e12fc5e92ca83780210b0cba78cb65207b582d131037e2b8bef9aa` | Hash of exact final CSV | BLOCKED | Checklist, notebook, report, evidence archive |
| Positive prediction count | TBD; historical 0.9112 file: 541 of 2,500 | Exact final CSV validation | BLOCKED | Checklist, ledger, report, notebook |
| Kaggle Public Score | Best verified benchmark: 0.9112; final-candidate score TBD | Kaggle submission history | BLOCKED | Checklist, READMEs, ledger, report, forum |
| Leaderboard position/timestamp | Unverified | Timestamped Kaggle leaderboard view | TODO | Checklist, Ukrainian report/forum only if dated |
| Source run ID | TBD | Final authoritative run artifact | BLOCKED | Checklist, report, notebook, status |
| Source config | TBD | Frozen tracked config plus resolved config | BLOCKED | Checklist, report, notebook |
| Final notebook path | TBD | Reviewed tracked notebook | BLOCKED | Checklist, READMEs, report, LMS |
| Final repository commit | TBD; creation baseline `d94967673d0278e294d175165bfb52065c1dc808` | Git after final checkpoint | BLOCKED | Checklist, READMEs/report if needed, LMS evidence |
| LMS upload timestamp | TBD | LMS receipt | BLOCKED | Checklist and evidence archive |
| Forum publication URL | TBD | Published forum page | BLOCKED | Checklist and public documentation if appropriate |

## 7. Documentation update map

Only `README.md` is being edited among these public/operational documents in the
current task.

| Document | Final update required | Dependency | Current action |
| --- | --- | --- | --- |
| `README.md` | Final candidate/result, optional screenshots, final status date | C01, C05, external evidence | Draft now; final values later |
| `README.uk.md` | Create Ukrainian academic overview with final values | C01 | Future file |
| `docs/final-project-report.uk.md` | Create detailed report/forum material | C01 | Future file |
| `docs/benchmarks/kaggle_submissions.csv` | Add/correct exact final filename, identity, count, and score | C05, C07 | Do not edit in current task |
| `docs/current-project-status.md` | Final checkpoint, completed work, remaining work, immediate next action | C01–E03 | Do not edit in current task |
| Final notebook | Frozen config, protocol, output identity, checks | C01 | Future file |
| Forum post | Final narrative, result, link, disclosure | D02–D04 | Future publication |
| This checklist | Every status, decision, evidence, and dynamic value | Continuous | Update as work progresses |

## 8. Final notebook requirements

The final notebook must be a **thin orchestration entry point**, not a monolithic copy
of the platform.

Acceptance requirements:

- readable narrative suitable for a university mentor;
- imports tested project modules;
- names and explains the selected final configuration;
- distinguishes AutoGluon screening, frozen candidates, and project-owned logic;
- validates all required inputs and identities;
- trains or executes the selected final workflow;
- uses fold-local preprocessing where evaluation requires it;
- refits the final model on all labeled rows when that is part of the frozen workflow;
- creates exactly 2,500 predictions;
- writes columns exactly `index,y` in sample-submission row order;
- validates binary labels, duplicate IDs, row count, and column order;
- reports threshold, positive count/rate, output path, and SHA-256;
- runs from a clean kernel using the documented environment;
- contains no absolute paths or usernames;
- does not depend silently on earlier notebook state, ignored predictions, or
  pre-trained models unless that dependency is explicit and permitted;
- produces the exact CSV claimed in the LMS/report evidence.

## 9. Submission definition of done

### Competition submission

`DONE` only when:

- dataset, candidate, seed(s), threshold, and blend are frozen;
- source run/config and artifact identities are recorded;
- the CSV has 2,500 rows and exact columns `index,y`;
- row order and binary predictions are validated;
- SHA-256 and positive count are recorded;
- the exact file is uploaded manually to Kaggle when applicable;
- Public Score and submission timestamp are copied from Kaggle without inference.

### LMS submission

`DONE` only when:

- the final notebook passes clean-kernel Run All;
- it reproduces the declared prediction artifact;
- visible outputs contain no private path or sensitive data;
- the exact reviewed notebook is uploaded once;
- the repository link is included;
- the submitted file and link are re-opened and verified;
- timestamped evidence is saved before 2026-08-02 23:45 Europe/Berlin.

### Public repository

`DONE` only when:

- README claims, results, links, and screenshots are reviewed;
- final values agree across English/Ukrainian docs, notebook, and ledger;
- no raw data, submissions, model binaries, secrets, or unintended local artifacts are
  tracked;
- no personal path remains in public-facing material;
- the final intended Git commit is recorded and the remaining status is understood.

### Forum publication

`DONE` only when:

- the Ukrainian report/post uses the frozen evidence;
- AutoGluon is disclosed accurately as auxiliary screening;
- manual/project-owned evaluation is described accurately;
- repository and any allowed result links work;
- publication URL and timestamped evidence are saved.

## 10. Deferred post-deadline work

Keep these items out of the critical submission path:

- consolidate legacy and Research v2 evaluation paths;
- remove or implement stale placeholders;
- generalize competition-specific row/schema assumptions;
- add generic classification, multiclass, and regression task contracts;
- make selected AutoGluon candidates first-class fixed Research v2 adapters where
  justified;
- execute additional full dataset/model campaigns;
- implement campaign-scale reporting views;
- implement generalized cross-dataset multi-model blending;
- consolidate submission and deployment entry points;
- clean large ignored artifact archives after preserving authoritative evidence;
- generate portable reports from validated artifacts;
- add stronger CI reproducibility checks;
- package services and environments with Docker;
- evaluate remote immutable artifact storage;
- create reusable project/competition templates.

## 11. Decision log

Append new rows; do not rewrite earlier decisions silently.

| Date | Decision | Rationale | Evidence | Affected files/tasks |
| --- | --- | --- | --- | --- |
| 2026-08-01 | Keep the public README in English | Serves mentors, engineers, recruiters, and future platform users | Finalization brief | A01–A05, D01 |
| 2026-08-01 | Prepare a separate Ukrainian academic README and report later | Keeps the public overview concise while supporting LMS/forum needs | Finalization brief | D01–D04 |
| 2026-08-01 | Use a thin final notebook | Reuses tested modules and avoids duplicating the platform in notebook state | Audit findings and finalization brief | C02–C06 |
| 2026-08-01 | Treat AutoGluon as auxiliary screening, not the entire project | AutoGluon use is permitted; screening must remain distinct from project-owned evaluation and deployment | Implemented environment/runner isolation and academic rule | B04–B06, C02, D01–D04 |
