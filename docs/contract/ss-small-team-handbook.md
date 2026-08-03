THE SMALL TEAM HANDBOOK
Operating Principles for Research Groups Building Models, Software, and Evidence
For: teams of three to five doctoral-level scientists — computer science, computational physics or biology, statistics, applied mathematics — who build models and software, answer analytical questions, and publish, while embedded in a larger group under a principal investigator and serving several concurrent external collaborations of differing skill, funding, priority, and goal.

Sources. This handbook borrows deliberately from professions that have had longer to think about small expert teams under uncertainty: high-reliability operations, site reliability engineering, aerospace mission operations, structural and mechanical engineering, clinical diagnostic reasoning, large-scale collaborative science, and the craft trades. Where a practice comes from somewhere specific, it is named, because knowing the source tells you where the practice stops working.


PREFACE: A ROLE WITHOUT A MANUAL
0.1 The position this handbook is written for
There is a role in modern research institutions that did not meaningfully exist twenty years ago and now exists in numbers: the permanent computational scientist. Not a trainee. Not a principal investigator. A doctoral-level scientist who stays close to the data, the model, and the code for the length of a career, works across many projects at once, leads a small team, and is evaluated on an ill-defined mixture of publications, infrastructure, service, and the success of other people's science.

The growth has been fast and largely unremarked. It is not unusual now for a computational department to have gone from one such position to more than a dozen inside a decade, with several of those people supervising teams of two to four. The role scaled because the alternatives stopped working: data volume outran what a rotating trainee could process, methods complexity outran what could be absorbed in a three-year postdoc, infrastructure required continuity that a trainee pipeline structurally cannot provide, and the supply of trained computational scientists began substantially to exceed the number of faculty positions.

The practice of the role has not scaled with the count. Nobody is trained for it. Everyone currently in it was trained for something else and adapted. There is no textbook, no curriculum, no shared vocabulary, and — critically — no accumulated craft knowledge being passed down, because the first cohort is still in it. That is the gap this handbook is written into.
0.2 Three shapes of a scientific career
The three roles overlap in technical skill and differ almost completely in everything else. Most of the friction people in the third role experience comes from being trained for the first and measured against the second.



Postdoc
Principal investigator
Career staff scientist
Time horizon
3–5 years, then leave
20–30 years
Indefinite, in place
Unit of work
The first-author paper
The grant and the person
The project, the method, the tool
Proximity to the artifact
Maximal, then ends
Declines every year
Sustained for a full career
Team
Themselves
Everyone, at a distance
2–5, closely, hands-on
Concurrency
One deep problem
A portfolio, abstractly
2–5 projects, concretely, at once
Depth trajectory
Deep, then reset in a new lab
Broad, increasingly delegated
Compounding in one direction
Authority
None, and none needed
Positional, real
Responsibility without authority
Optimizing for
A job
The group's survival and direction
Being the person who can actually do it
Failure mode
Not getting the job
Losing touch with the work
Drift, dilution, and invisibility


0.2.1 The postdoc is a solo operator on a clock. The structure of the position — temporary, individually credited, judged on first-author output — makes solo depth the rational strategy. Everything the postdoc learns about how to work is learned in a configuration they will never occupy again if they succeed. They are also, in most cases, training for a job they will not get, which is a separate problem but colors everything.

0.2.2 The PI's skill migrates away from the artifact. This is not a criticism; it is the job. Their comparative advantage moves from doing to selecting, funding, positioning, recruiting, and defending. A PI five years in reads less code than they did; ten years in, considerably less; twenty years in, often none. The judgment they retain about what matters is enormous and their contact with what is actually true in the data becomes mediated entirely by other people.

0.2.3 The staff scientist is the person for whom neither of those is true. They stay on the artifact. Their depth compounds rather than resetting. They hold several problems simultaneously, in detail, and they are the connective tissue between what the PI intends and what is computationally the case. They are also — and this is the part nobody warns you about — running a team, managing a portfolio, doing cross-disciplinary translation, maintaining infrastructure, and developing junior people, none of which they were trained for and most of which is invisible in the metrics they are assessed by.
0.3 What the training gives you and what it doesn't
A PhD and postdoc produce, reliably: technical depth, the ability to work alone on a hard problem for years, tolerance for failure, and taste in a narrow domain. These are real and they are the foundation.

They do not produce, and are not designed to produce:

Portfolio judgment — choosing among five viable projects with different funders, timelines, and collaborators, and knowing which to decline
Concurrency discipline — holding several problems in working memory without any of them silently degrading
Working through people without authority — the single largest time expenditure in the role and the one with no training pathway at all
Cross-disciplinary translation — the daily work of being the only person in the room who understands both the assay and the inference
Triage under simultaneous load — what to drop when four things escalate the same week
Standards and infrastructure — the unglamorous, uncredited work that determines whether anything is reproducible in three years
Developing other people — the thing that determines whether your capacity scales or plateaus at your own two hands
Stopping — recognizing a dead project and killing it, which is far harder when it is one of five than when it is your only one

Every chapter after this preface is about one of those.
0.4 The operator problem
There is a useful idea in the professions that maintain small teams of highly trained specialists: the distinction between people who do the thing and people who manage people who do the thing. Most career structures force a transition from the first to the second, usually around the point where the person has finally become good at the first.

Science has the same structure and the same loss. The conventional path removes you from the work at exactly the moment your judgment about the work becomes valuable.

The career staff scientist role is the exception, and that is its whole point. It is the position that lets a scientist remain operational — hands on the data, close to the ground truth, still capable of noticing that a result is wrong for a reason nobody has articulated — for two or three times as long as the conventional path allows. Institutions have built these positions because they discovered they needed people with fifteen years of accumulated pattern recognition who are still willing to open the file.

That longevity is the asset. It is also the thing most easily squandered, because the role has no natural defenses:

Nobody schedules your infrastructure work, so it doesn't happen
Nobody limits your project count, so it grows
Nobody enforces your exit from a collaboration, so you never leave one
Nobody protects your unassigned time, so it is the first thing consumed
Nobody tells you when a project is dead, because everyone else has less information than you do

A postdoc's structure is supplied by their deadline. A PI's structure is supplied by their grants and their committee. A staff scientist's structure has to be supplied by the staff scientist. That is the actual argument for this handbook.
0.5 Latitude without management
The good version of this role is largely unmanaged. You are given intent — the themes of the grants, the direction of the program — and considerable latitude to explore, choose methods, and publish within it. This is correct, and it is why the position attracts people who would be miserable elsewhere.

It also means the entire middle layer is missing. Nobody is doing sprint planning for you. Nobody is going to notice that three of your five collaborations are in the same failure mode. Nobody will tell your team that the pipeline needs to be fixed before the next analysis. The latitude and the absence of structure are the same fact, and the difference between a group that thrives on it and one that dissolves into it is entirely whether the internal structure gets built.

The specific things that must be supplied internally, because no one else will: planning discipline (Part III), standard responses to common failures (Chapter 16), a triage order (Chapter 21), stopping criteria (Chapter 25), coverage against absence (Chapter 27), and the conditions under which junior people will tell you something is wrong (Chapter 32).
0.6 Responsibility without authority
You will be responsible for outcomes across projects where you can direct no one. You cannot compel a clinician to standardize metadata, an experimentalist to change a protocol, a core to change a format, or a collaborating PI to prioritize your analysis. Your entire toolkit is credibility, competence, relationship, and making the correct path the easiest one.

This is genuinely difficult and it is the reason Chapter 30 is longer than Chapter 4. It is also the strongest argument for the handbook's central bias: when you cannot direct, you must agree in advance. Every stopping rule, data agreement, authorship criterion, support-intent statement, and exit condition in these pages exists because a written agreement made calmly at the start is the only enforcement mechanism available to someone with no positional power.
0.7 Why three to five
Three to five expert people is a strange organizational size. It is too small for process and too large for telepathy. It is the size at which everything is coordinated informally right up until the moment it isn't — and when it fails, the failure is always attributed to the specific thing that broke rather than to the absence of structure.

It is also close to optimal. Coordination cost grows faster than headcount, and five is roughly the maximum size at which everyone can still hold the whole problem. The teams described here are small on purpose, and the practices in this handbook are calibrated to that size. Some of them do not scale past it, and where that is true it is noted.
0.8 What this handbook will not do
It will not make the work urgent. Almost no analytical decision is time-critical, and most are reversible. Structure exists to protect thinking, not to accelerate it past the point of correctness. This matters especially in a role with no external clock: without a deadline forcing the pace, manufactured urgency is entirely self-inflicted and entirely optional.
It will not replace taste. Chapters 2, 10, and 46 are about the parts no process can supply: which problems matter, what an interesting result feels like, and how to stay curious for thirty years. In a career measured in decades rather than grant cycles, these are the compounding assets. Everything else is scaffolding around them.
It will not treat rigor and creativity as opposites. They are the same faculty pointed in different directions. A team that cannot generate strange ideas has nothing to be rigorous about; a team that cannot be rigorous cannot tell which strange ideas were right.
It will not survive being followed literally. Every chapter should be modified by your team within a year. If it hasn't been, nobody is reading it.
0.9 How to use it
Read the Preface, Chapter 2 (the ethos), and Chapter 52 (what not to borrow) first. Those three establish what the rest is for.

Then take one thing. The highest-return single practices, roughly in order: the project charter in Appendix B; the coverage matrix in Chapter 27; sketching figures before analysis (14-1); the go/no-go poll (13-5); and stop-the-line authority (32-1). Each takes under an hour to institute and each addresses a failure mode that will otherwise recur indefinitely.

Use the rest as reference. It is a handbook, not a program.


PART I — FOUNDATIONS
Chapter 1: Five Truths
Goal. Establish the constraints you cannot engineer around. Where the wisdom comes from. Small-unit doctrine, where the equivalent list describes how a force is built rather than how it fights. What you should walk away with. Five sentences you should be able to recite, and the knowledge that violating any of them is expensive rather than merely untidy.

These are constraints, not goals. You cannot engineer around them; you can only plan for them.

1. People matter more than infrastructure. Correct problem framing beats a larger compute allocation, every time. The bottleneck is almost never hardware, and a team that believes it is will keep buying capability it cannot use.

2. Quality beats quantity. One well-characterized dataset with clean, complete, trustworthy metadata beats five poorly annotated ones. Additional sample size does not repair a confounded design; it makes the wrong answer more precise.

3. Expertise cannot be produced on demand. A person who holds domain science, statistics, and software architecture simultaneously took a decade to make. You cannot hire your way out of a methods problem on a project timeline.

4. Capability must exist before it is needed. The pipeline built during a deadline panic is the pipeline that fails. Infrastructure, standards, and skill are built in quiet periods or they are not built at all.

5. Most of the work is not yours. The analysis is the visible ten percent. Without the experimentalists, the clinicians, the cores, the data managers, and the engineers, there is nothing to analyze. The people whose names appear in the middle of the author list generated the thing the paper is about.

The fifth truth is the one that computational teams forget most reliably, and forgetting it is the most common cause of collaborations that end badly.
Chapter 2: The Scientist's Ethos
Goal. Describe the disposition the rest of the handbook exists to serve. Where the wisdom comes from. Nowhere operational — this is the one chapter with no borrowed source, because operational professions are handed their objectives and science is not. What you should walk away with. A set of habits that keep curiosity alive across decades, and the recognition that these are the scarce faculties rather than the soft ones.

Everything else in this handbook is machinery. This chapter is the thing the machinery is for. It is also the part that cannot be borrowed from any operational profession, because operational professions execute known objectives and science does not.

2-1. The disposition is inquisitiveness, not productivity. Productivity is what a team has when it knows what to do. Inquisitiveness is what generates something worth doing. A group optimized purely for throughput will produce a great deal of competent, uninteresting work and will not notice.

2-2. Anomaly appetite is trainable, and it is the core skill. The default human response to a result that doesn't fit is irritation — it means more work, a broken timeline, a complication. The scientific response is appetite. Train yourself and your people to feel a small lift, not a small sinking, when something doesn't fit. Practically: when someone reports an anomaly, the first response in the room should be interest, never "can we work around it." The room learns what to bring you within about three instances.

2-3. Cultivated confusion. The ability to remain in a state of not-knowing without prematurely collapsing to an answer is a professional skill and it is uncomfortable. Most bad science is not produced by people who reasoned incorrectly; it is produced by people who could not tolerate the discomfort of an open question and resolved it early. Name this in the group. Give people permission to say "I don't understand this yet" for longer than feels acceptable.

2-4. Keep a live question. Have a problem in your back pocket at all times — something you are not funded for, not assigned to, and cannot currently solve. This is what allows an unrelated paper, talk, or dataset to become useful; a prepared mind is one that is currently carrying a question. Teams should each maintain a visible list.

2-5. Distinguish interesting from important. These gradients diverge, and following interest alone produces a career of clever irrelevance. Ask periodically: what are the important problems in my field, and why am I not working on one? The answer is often good. It should not always be.

2-6. Protect unbudgeted exploration. Some fraction of time — name it, defend it, put it on the calendar — should have no deliverable. Not "innovation time" as a slogan; a real, small, protected allocation that nobody has to justify. This is the single most commonly promised and least commonly delivered practice in research groups.

2-7. Taste is real and it is learnable only by exposure. The sense of which results matter, which methods will age well, which papers are load-bearing — this is not teachable by instruction. It is transmitted by working alongside people who have it, watching what they attend to and what they dismiss. Structure your group so juniors see seniors choosing, not just executing.

2-8. Analogy is the engine. Nearly all methodological novelty is transfer: a structure from one domain applied to another. This means reading outside your field is not a luxury, it is the input to the generator. Schedule it. One paper a week from somewhere you have no business being.

2-9. Constraints generate. Given unlimited resources, most teams produce a slightly larger version of what they already do. Artificial constraints — solve it with one figure, solve it with a linear model, solve it in a day — reliably produce different ideas. Use them on purpose.

2-10. Being wrong is cheap; being uninteresting is expensive. A wrong result, honestly obtained and honestly corrected, costs you a little credibility and teaches the field something. An uninteresting result costs you years. Calibrate risk accordingly; most academic teams are far too conservative.

2-11. Generosity compounds. Give away ideas you cannot execute. Tell people what you are working on. The received wisdom about scoop risk is wildly miscalibrated relative to the returns on being the person everyone thinks with.

2-12. Writing is thinking, not reporting. If you cannot write the paragraph, you do not have the result. Start writing in week two, not month ten. The gaps in the prose are the gaps in the work, and they are visible much earlier in writing than in code.

2-13. Elegance is a real signal and a real trap. Simple, beautiful explanations are more often right than ugly ones, and this is a legitimate heuristic. It is also how people fall in love with models that are wrong. Use elegance to generate hypotheses; never use it as evidence.

2-14. Protect the naive question. The most valuable sentence in a group meeting is "sorry, why do we do it that way?" It is also the sentence most easily suppressed by seniority, and once suppressed it stops being offered. Ask it yourself, in public, on purpose, so that others can.
Chapter 3: Operating Principles
Goal. Give a short list of principles for how work is chosen, planned, and carried, plus five you can recall while tired. Where the wisdom comes from. Operating doctrine of small expert units that work through partners rather than through authority. What you should walk away with. Around a dozen principles, of which the ones about working indirectly and about credibility do most of the work.

Twelve principles for how work is chosen, planned, and executed. Adapted from the operating doctrine of small expert units that work through partners rather than through authority.

3-1. Understand the environment before you shape it. Not just the dataset — the field. Who else works this problem, who reviews it, what is fundable, what your collaborators actually need, and who makes decisions on both sides. Most methods work dies from environmental misreading, not technical failure.

3-2. Build for the situation you did not plan for. You cannot write a procedure for every case. A team that understands why the analysis exists adapts correctly when 40% of the data fails QC. A team with only the procedure stalls and waits.

3-4. Coordinate across organizations you do not control. Cores, sequencing, pathology, statistics, IT, legal, the collaborating lab. Unity of effort among people who do not report to you or to each other is the actual job description.

3-5. Choose engagements deliberately. You are the scarce resource. Every project taken is a project foregone. Not every problem deserves your direct effort — many are better deferred, routed elsewhere, or answered with a two-hour conversation instead of a two-month analysis.

3-6. Weigh long-term against short-term effects. The quick expedient pipeline that produces this week's figure is the pipeline three groups depend on in two years. Short-term wins that damage long-term position are losses on a delay.

3-7. Credibility is the asset. Reproducibility, honest negative results, not overselling. It takes years to build and one paper to spend. Everything else on your balance sheet is downstream of it.

3-9. Work through people, not around them. ★ The highest-leverage principle here. Your effect is measured in capability transferred, not analyses delivered. The collaborator who can now run their own analysis is worth more than ten analyses you ran for them — to them, to the field, and to your own capacity. See Chapter 30.

3-10. Keep a second path warm. Maintain more than one viable approach, and be able to switch before or during execution. Three enablers: contingency plans tied to specific decision points; genuine rehearsal; and the same people plan, rehearse, and execute. That last one is a prohibition on handing a specification to someone who was not in the design conversation.

3-11. Do not start what you cannot sustain. An abandoned package with four hundred users is a negative contribution. A pipeline that requires your machine, your environment, and your presence is a dependency you created and will service forever.

3-12. Balance protection against coordination. Over-restricting access excludes the people who would have made the work correct. Under-restricting can cost priority or violate governance. Both failures are real; neither is the default answer, and the choice belongs in the plan.

3-13. Reduce uncertainty before committing. Literature, prior cohorts, pilot data, the actual QC report — not somebody's summary of the QC report. Cheap information first, expensive commitment second.
The five daily principles
Short enough to hold in working memory. Adapted from small-unit operating doctrine, where the point of a five-item list is that it can be recalled while tired.

3-14. Plan simply and communicate completely. A great plan that takes six weeks and that nobody has read is not a great plan. Speed of dissemination beats sophistication.

3-15. Verify what you think you know, and find out what you don't. This is the definition of exploratory analysis. Half of EDA confirms assumptions you already hold. That half is not wasted — it is the primary purpose.

3-16. Protect the irreplaceable. Raw data, provenance, and people. Everything else can be regenerated. Burnout is attrition and it does not reverse on the project timeline.

3-17. Concentrate. Clear intent plus disciplined communication, so that everything available lands on the decisive question. Five half-analyses lose to one that is fully resourced.

3-18. Use judgment. Do the thing you are supposed to do without being told, despite discomfort. The re-run you don't want to do is usually the one that matters.
Chapter 4: Leading Through Intent
Goal. Describe how a small team runs when nobody is issuing orders. Where the wisdom comes from. Mission command doctrine, adapted for a group with no positional authority. What you should walk away with. The seven conditions that make decentralized work possible, the structural roles most labs never name, and the reason a lead who is still coding full time caps the team.

4-1. Seven conditions for decentralized work. Competence, mutual trust, shared understanding, clear intent, latitude in method, disciplined judgment, explicit risk acceptance.

Competence — the amount you can decentralize is bounded by demonstrated capability. Trust is calibrated, not granted, and calibrating it honestly is a kindness.
Mutual trust — built on shared experience; there are no shortcuts. Critically, people exercise judgment only when they believe the leader will support the outcome of their decisions. ★ Punish one good-faith wrong call and initiative ends, permanently, and you become the bottleneck for everything.
Shared understanding — the dominant failure mode in cross-disciplinary work. Part VII.
Clear intent — state the question and the end state, not the steps.
Latitude in method — specify what and why; leave how. "Determine whether X precedes Y in this cohort, to a standard that survives review" beats a numbered protocol and produces better work.
Disciplined judgment — follow the plan until you recognize the plan no longer fits the situation you're actually in. ★ The hard skill: knowing when to abandon the pre-registered analysis, plus the discipline not to do it merely because you dislike the answer.
Risk acceptance — distinguish risk to the project (this may not work) from risk to the group (this may be wrong and we may publish it). The second is categorically different and belongs to the PI.

4-2. Structure of a small team.

Role
Owns
PI
Problem selection, resourcing, external position, decisions that risk the group
Team lead / staff scientist
Translating intent, technical direction, external coordination, allocation
Standards owner
Code standards, QC thresholds, reproducibility, release quality — how, and to what standard
Specialists (2–3)
Methods, implementation, analysis, with deliberately overlapping coverage


4-3. The standards role is the one nobody names. High-reliability organizations separate what and why from how and to what standard, and staff them differently. Almost no research group does this, which is why code standards, QC thresholds, and reproducibility practice end up being set by whoever cares most that week. Name the role. It does not have to be the most senior person; it should be the most exacting one.

4-4. Every decision has exactly one owner. Ambiguous ownership between a computational lead and a collaborating PI is the most common cause of stalled multi-group projects. Write the owner's name in the plan.

4-5. Know two levels in both directions. Your PI's priorities and pressures; your rotating student's actual blocker. Continuously, not quarterly.

4-6. Designate succession before you need it. Who runs each project if you are unavailable for a month? If the answer is "nobody," that is the finding.

4-7. Lead from one level above the work. A team lead absorbed in the code has stopped leading. The staff scientist who remains both the best programmer in the group and a full-time programmer has a structural problem that caps team throughput permanently.


PART II — CHOOSING AND SHAPING WORK
Chapter 5: Kinds of Work
Goal. Separate the kinds of work a computational team is asked to do, because each has a different tempo and stopping rule. Where the wisdom comes from. Special operations core activities, which exist for the same reason — different mission types need different planning. What you should walk away with. The habit of naming which kind of work a request is before planning it.

Different kinds of work require different planning, tempo, and stopping rules. Small teams routinely apply publication-grade rigor to feasibility questions and feasibility-grade rigor to deliverables. Name the kind before planning.

5-1. Bounded delivery. A specific, defined output for a specific requester. Defined objective, defined end, planned handoff. Most collaboration requests are this and get mismanaged as open-ended research.

5-2. Reconnaissance. Feasibility work and pilots. The deliverable is information for a decision, not a result. Reconnaissance that silently becomes delivery is the most common form of scope creep in computational groups.

5-3. Capability building. The tool, method, or framework that lets a whole community do something new. Highest leverage, longest timeline, hardest to fund, worst short-term optics, best long-term return.

5-4. Advisory work. Supporting another group's capability so they can eventually operate without you. ★ The default mode for most collaborations, and the one whose success criterion is most often mis-set. See Chapter 30.

5-5. Emergency response. A reviewer's fatal objection, a bug discovered in published work, a competitor's preprint. Short, intense, disruptive to everything else, and legitimately preemptive.

5-6. Institutional work. Committees, cores, data governance, teaching, infrastructure, relationships. Universally undervalued in performance review and universally determinative of whether anything else is possible.

5-7. Dissemination. Talks, preprints, documentation, tutorials. Part of the work, not overhead on it.
Chapter 6: Modes of Engagement
Goal. Give a vocabulary for how a given project is fought, once you know what kind it is. Where the wisdom comes from. Offensive and security operations doctrine. What you should walk away with. Twelve named modes, of which planned-exit engagements and routing around obstacles will save you the most time.

Once you know the kind, choose the mode. Each has a characteristic tempo, resource profile, and exit.

6-1. Exploratory entry. Move into unfamiliar data to find out what is there. The rule: make first contact with the smallest possible commitment. One sample, one chromosome, one field of view, a 5% subsample. Do not commit the full pipeline to the first look.

6-2. Rapid response. Minimal preparation, executed to exploit a window — a grant deadline, a collaborator's committee meeting. Legitimate, and it must be declared as such, in writing, so nobody downstream mistakes the output for a validated result. Most misuse of preliminary data begins with an undeclared rapid response.

6-3. Deliberate build. Full planning, verification, rehearsal, documentation. The methods paper, the released pipeline. Different standard, different timeline, different artifacts. ★ Confusing 7-2 and 7-3 is the most common tempo error in computational groups, in both directions.

6-4. Bounded engagement with planned exit. A defined piece of work for an external group where the exit is designed at the start: what you deliver, when you leave, what you hand over, what you explicitly will not maintain. Engagements without a designed exit become permanent unfunded support obligations, and this is the single most common way a computational team loses its capacity.

6-5. Prepared opportunism. You have a method ready and you wait for the dataset, the gap, or the opening. Requires having something in reserve, which is why most teams cannot do it — everything they have is deployed.

6-6. Quiet entry. Work a problem without announcing it, entering a crowded field by the route nobody is defending. If you are noticed early and the incumbents are better resourced, disengage rather than compete on their ground.

6-7. Supporting role. You enable someone else's result without owning the objective. ★ The requirement is to specify the intent of the support at the outset. Are you delivering a definitive analysis, keeping one question open, removing a single obstacle, or providing continuous support to their end point? Unspecified support obligations expand until they consume the team, and the acknowledgment section is where they are repaid.

6-8. Critique. The benchmark, the reanalysis, the negative evaluation. You defeat a position without building the replacement. Legitimate, sometimes the highest-value contribution available, and worth being honest that it is what you are doing.

6-9. Routing around. Fix the obstacle you can clear in ten minutes; go around the one that will take three days, and log it. ★ The instinct to solve every obstacle encountered is the largest single source of lost tempo in computational work.

6-10. Pressing an advantage. When something works, immediately: second dataset, second application, follow-up, tutorial, talk. Most academic teams fail here. They reach the objective and stop, leaving the ground they opened for someone else to occupy.

6-11. Defending a position. Protecting a published method against critique. Prepared defenses — documentation, benchmarks, reproducible examples, a public test suite — beat improvised ones by an enormous margin.

6-12. Holding. Keeping a project alive at minimal resource until conditions change. A legitimate mode only with a stated resumption condition: a funder, a dataset, a person, a dependency. Holding without a resumption condition is neglect with better branding.
Chapter 7: Accession
Goal. Describe the intake gate through which every dataset passes, and why it lands on your most experienced people. Where the wisdom comes from. Archival accessioning practice, which has spent a century deciding what to take in and how deeply to process it. What you should walk away with. A tiered, default-shallow intake process; a written assessment that is a deliverable rather than a byproduct; and the habit of counting the datasets you recommend against.

Every dataset enters the group through this chapter. It is simultaneously the largest uncompensated cost a computational team carries and its least-exercised source of leverage.

7-1. What accession is. The assessment performed on a new dataset — internal or external — before any project exists: quality control, metadata curation, reconstruction of the experimental design, sample tracking, storage assignment, selection and execution of standard pipelines, and a judgment about what questions the data can actually support. It is not analysis. It is the determination of whether analysis is possible and what kind.

7-2. Why it lands on the most experienced people. The expertise asymmetry is larger here than anywhere else in the portfolio. An experienced person assesses an external dataset's design in two days. A second-year student takes six weeks and reaches a wrong conclusion that nobody catches — and this is the one place in the whole pipeline where a wrong conclusion has a multi-year half-life. ★ A mischaracterized design, a missed batch variable, or a misread sample manifest becomes the inherited premise of every downstream analysis. This is Chapter 46-3 operating at the point of entry: the provisional assessment made in week one is ground truth by month six, and nobody re-derives it, because by then it is simply "how the data is."

7-3. Accession is a formal act, and declining is one of its outcomes. Borrowed from archival practice, where accepting a collection is a discrete decision with a recorded rationale, a condition report, assigned identifiers, and assigned storage — and where declining to accession is a normal, documented outcome rather than a failure. Most computational groups perform the assessment while treating the acceptance as having already happened. Separate the two. The assessment produces a recommendation; someone with authority accepts or declines.

7-4. Default to minimal processing. Around 2005 the archival profession, confronting backlogs it could never clear, concluded that processing everything deeply meant nothing became accessible, and adopted a principle of minimal processing by default with deep processing reserved for material that demonstrates it warrants the effort.

Adopt the same tiering:

Tier
Trigger
Depth
Shallow (default)
Any new dataset
Standard QC report, metadata skeleton, storage assigned, one standard pipeline, two-page assessment. Days, not weeks.
Standard
A specific question has been posed
Design reconstruction, metadata harmonization to the group schema, pipeline selection with rationale, feasibility verdict
Deep
The project has cleared a phase gate (Ch. 8)
Full curation, custom preprocessing, publication-grade provenance, deposit preparation


★ Most datasets never leave the shallow tier, and effort spent curating them to publication standard is unrecoverable. Deep curation is earned by the project, not granted at arrival.

7-5. The assessment report is a deliverable, not a byproduct. It is currently being produced and thrown away. Give it a fixed format, a version, and an author:

Experimental design as understood, with confidence flagged per element
QC summary and failure modes observed
Metadata completeness, gaps, and where the gaps must be filled from
Sample tracking: what physically exists, where, and under what identifiers
Storage: raw location, working location, projected footprint and cost
Pipelines run, versions, and rationale for selection
Known problems and specific risks to downstream inference
Feasibility verdict: what questions this data can and cannot support
Recommended next phase, or recommendation against

Two consequences follow. It becomes the artifact you hand back when you decline to go further, which converts a refusal into a delivery. And it is the natural seed of a data descriptor paper — the one real publication path in this product line and almost universally unused.

7-6. Data contracts with providers. Specify what must arrive before assessment begins: sample manifest in a named format, written design description, protocol version, batch and processing structure, consent and governance status, checksums, and a named contact who can answer questions. Nothing enters the queue until the contract is satisfied. This is standard practice in data engineering and it relocates work to the party who actually holds the information, which is both correct and the only version that scales.

7-7. Count the declines. Clinical trials report screen failures as a number, not as waste. Your assessments that conclude this data cannot answer that question are among the highest-value work the team does and are currently invisible, because a project that does not happen leaves no trace. ★ A line in the quarterly report reading "assessed 14 datasets, recommended against 5, estimated four person-years of downstream effort avoided" will change how the department understands the team more than another manuscript would.

7-8. Split execution from assessment. The boundary with a core facility, service group, or engineering team runs here and it is clean:

Delegable — storage provisioning, sample tracking, checksum verification, standard pipeline execution, format conversion, archival deposit, monitoring. Identical every time.
Not delegable — experimental design interpretation, feasibility judgment, choosing which pipeline is appropriate and why. These require the pattern library and they are the whole value of the step.

7-9. Deaccession. You are storing datasets that will never be published and are consuming money and attention. Have a documented retirement procedure: reduced storage tier, archival deposit, or deletion with a record of what was deleted and why.

7-10. Fund it as infrastructure. Accession happens before a project exists, which is precisely why no grant pays for it and why it is invisible. ★ The argument to make: accession is not overhead on projects — it is the mechanism by which the group decides which projects exist. It belongs on a departmental or program line. Funded that way, the queue becomes governable; funded project-by-project, it never will be.

[EXAMPLE — TODO] Placeholder, replace with your own: "An external cohort arrived with a manifest whose sample IDs did not match the FASTQ filenames, and the treatment timepoints turned out to encode two different conventions depending on which site collected them — two days of assessment saved a year of analysis built on a wrong design."

Public case that fits. In the Duke chemosensitivity affair, biostatisticians at MD Anderson attempting to reproduce published predictors found shifted data columns, switched sample labels, and cases where the same samples appeared as both drug-sensitive and drug-resistant. Correcting the errors made the correlations disappear. The predictors had already been used to assign patients to treatment in three clinical trials, which Duke terminated in 2010. Baggerly & Coombes, Annals of Applied Statistics 3(4):1309–1334 (2009); summary at https://www.ncbi.nlm.nih.gov/books/NBK475955/
Chapter 8: Engagement Tiers and Flow Control
Goal. Solve the problem that success escalates an engagement and nothing displaces it. Where the wisdom comes from. Architectural phase-gating and air traffic flow management. What you should walk away with. Four engagement tiers, the displacement rule, and the discipline of queueing work at intake rather than in the middle.

8-1. The problem this chapter solves. A service group has a scope ceiling supplied by its catalogue: a defined menu at a defined depth, with anything beyond it declined and routed elsewhere. A research team embedded among independent investigators has no such ceiling, and worse, success raises the floor. A good result on a first-pass exploration does not conclude the engagement — it escalates it, because now people care and want the manuscript. You cannot decline that escalation, because the escalation is both the value of the model and the thing you are evaluated on.

★ Escalation-by-success is not the problem; it is the entire advantage of this structure over a service core. The problem is that escalation currently happens silently and additively — the new manuscript project does not displace anything, it stacks.

8-2. Four tiers.

Tier
Commitment
Ends with
0 — Consultation
Hours. Open to anyone.
A recommendation, not an analysis
1 — Exploration
1–3 weeks, declared and bounded
A written finding and a three-way decision: stop, hand back, or escalate
2 — Analysis
Defined deliverable, defined end
The deliverable and a documented exit
3 — Co-development
Open-ended, toward manuscript
Authorship agreed at entry; owner named; PI signs off


8-3. Phase gates and re-contracting. Architecture and engineering consultancies span exactly this spectrum — feasibility study through schematic design through full documentation through construction support. They manage it not with a scope ceiling but with explicit re-contracting at each phase boundary: nothing slides forward, and each transition is a decision with new terms, new fee, and new resource commitment. The scope is unbounded and controlled. That is the mechanism to import.

8-4. The displacement rule. ★ Nothing enters Tier 3 without something leaving it. This is the gate a service core gets from billing and that you must get from arithmetic. It also converts every escalation into a conversation about tradeoffs rather than capacity, which is a better conversation and one where you are not the person saying no.

8-5. Demand-capacity management, from air traffic control. ATC solves exactly your problem — more aircraft want to land than the runway can accept — and solves it without heroism, at national scale, continuously. Three transferable mechanisms.

Published capacity. Every airport and every control sector has a published maximum capacity. When projected demand exceeds it, measures are taken to reduce traffic, and this is a named, routine, non-scandalous procedure — regulation. ★ The entire concept depends on capacity being published in advance, as a number, before it is exceeded. A team whose capacity is unstated is a team whose capacity will be exceeded and then blamed.

A monitored alert threshold. The FAA's Monitor Alert Parameter is a numerical trigger that notifies facility personnel that a sector's efficiency may degrade during a specific period. Alerts are graded — a lower tier warns that service quality will be degraded, a higher tier demands action — and they are logged, retained, and subjected to post-event analysis. The point is not the alarm; it is that overload is a measured, recorded quantity rather than a feeling somebody has on a Friday.

Set your own. Concurrent Tier 3 engagements per person. Total FTE committed versus available. Weeks of queued accession. When the number crosses, it triggers a defined action, logged, and reviewed later.

Absorb the delay at the gate, not in the air. A Ground Delay Program holds aircraft at their departure airport when arrival demand will exceed capacity, because if delay is unavoidable it is far better absorbed on the ground than in a holding pattern — safer and cheaper. ★ This is the single most useful transfer in the chapter. A project you have not started can wait indefinitely at almost no cost. A project you have started and cannot resource burns fuel continuously: context reloading, collaborator anxiety, half-built infrastructure, meetings about why it is stalled, and reputational damage that a delayed start would never have incurred. Queue at intake. Never in the middle.

8-6. Publish the queue, with the delay attached. ATC flow management is collaborative: carriers can see the program, understand the assigned times, and substitute among their own slots. Adopt the same transparency. A visible list of engagements with tier, owner, FTE, and expected start converts an invisible personal capacity problem into a shared allocation problem — and lets collaborators reprioritize their own requests against each other, which they will do more sensibly than you can do for them.

8-7. The intake one-pager. Every request, regardless of size, answers: the question in one sentence; what data exists and its accession status (Ch. 7); the requester's deadline and what it is driven by; which tier is being requested; who on their side does the work; and what happens if the answer is no. Ten minutes to fill in. Most low-value requests do not survive the last question.

8-8. Route the escalation, do not absorb it. When a Tier 1 exploration produces something people want to publish, the correct response is not to continue quietly. It is to stop, write the finding, and re-contract: what tier, what displaces it, what authorship, what timeline. The silent slide from Tier 1 to Tier 3 is how a team of four ends up carrying ten open manuscripts.

8-9. Equity is part of the mechanism. Flow programs are designed so delay is distributed by a stated rule rather than by who complains loudest. Publish your prioritization criteria — strategic fit, feasibility, resourcing, deadline externality — and apply them visibly. A group without stated criteria allocates by proximity and volume, and everyone learns to escalate socially.
Chapter 9: The Shape of the Ask and the Shape of the Data
Goal. Handle the two axes that vary independently at intake: how well-formed the request is, and how much signal the data contains. Where the wisdom comes from. Clinical reasoning on premature closure, plus equivalence-testing statistics. What you should walk away with. The reversibility question, the three-questions-ranked response, and the rule that nulls are where analytical flexibility is most dangerous.

Everything in Part III assumes the question precedes the data. In practice you receive every combination, and two axes vary independently: how well-formed the request is, and how much signal the data contains. Diagnose both at intake, because the failure modes are different and two of the four combinations are actively dangerous.
The shape of the ask
9-1. The two pathological extremes look opposite and share one defect. "Prove that X equals Y" and "give me a paper, I don't care what" both arrive without a claim that could come out either way. One has a conclusion and no question; the other has neither.

9-2. "Prove that X equals Y." The conclusion arrived before you did. You are being cast as an instrument rather than a scientist, and the social cost of returning X ≠ Y has been quietly transferred to you.

Three responses, in order:

The reversibility question. "What would it mean if the answer is no?" ★ If there is no answer, or the answer is that it cannot be no, you have diagnosed a confirmation request rather than a question — at intake, when it is still negotiable, rather than at month eight when it is not.
Convert demonstration into test, in writing, before looking. "We will test whether X equals Y, with equivalence margin D, in n samples, and we will report the result either way." Agreement to that sentence is socially easy beforehand and nearly impossible afterward. Chapter 52 applied at intake.
Make the statistical point, which is usually the most useful thing you can say. "Prove X equals Y" is a request to prove a null, which conventional testing cannot do. It requires equivalence testing with a pre-specified margin, or a Bayesian framing with a stated region of practical equivalence. Telling a collaborator "you have to tell me how similar counts as the same" moves them from assertion to specification, and it does not sound like refusal.

9-3. Take the confirmation request seriously as a hypothesis. Sometimes they are right, and they are holding unpublished observations, pilot data, or clinical intuition you do not have. "What makes you confident?" is worth asking every time; it frequently improves the design and it costs nothing.

9-4. "Give me a paper, I don't care what." This is a request for you to supply the scientific question, which is the most valuable thing your team does, and it is being priced at zero.

The conversion: return three questions the data could answer, ranked, with a one-line feasibility note each, and make them choose. This bounds an unbounded request, puts the choice where the domain knowledge lives, and gives you their commitment to a specific claim. ★ If they cannot choose, that is the finding — a collaborator who will not pick a question will not defend one in review either.

9-5. Price question generation correctly. If you supply the question, the design, and the analysis, that is Tier 3 (Ch. 8-2) with authorship agreed at entry, not a favour. This is the single most common place where a team's intellectual contribution goes unrecorded.
The shape of the data
9-6. Too much signal is a sequencing problem, not an abundance problem. A dataset containing ten papers has four characteristic failure modes: one sprawling paper that does everything badly; ten thin ones; never finishing, because every analysis opens three more; or being scooped on the interesting result while finishing the boring one.

Order by these questions, in this priority:

Which one, published first, makes the others easier? Usually the resource or descriptor paper that establishes the cohort, the annotation, and the method. It is the least interesting one. Publish it first anyway.
Which is most likely to be scooped? Time-sensitivity outranks interest.
Which can you give away? ★ A dataset with ten papers in it is collaboration currency. You cannot write ten. Handing three to collaborators with authorship buys more than writing all ten badly.

9-7. Write the boundary statement before the analysis. "This paper answers Q1 and explicitly does not address Q2 through Q10, which are named in the discussion." Scope creep in rich data is close to unstoppable without a written boundary, because every excluded question feels like an omission rather than a decision.

9-8. Rich data multiplies your accession risk by the number of papers. When ten manuscripts descend from one dataset, the week-one annotation is inherited by all ten without re-derivation. Chapter 47-3 compounding at scale. Budget deep accession (Ch. 7-4) accordingly — this is the case where it is worth it.

9-9. No signal: separate three things the collaborator will hear identically. No effect, no power, and no effect detectable with this design are different findings with different consequences, and conflating them is how a design flaw gets reported as biology.

9-10. Run a positive control before reporting a null. Does the pipeline recover something you know is present — sex-linked expression, a spike-in, a documented treatment effect, a technical gradient? ★ If the known signal does not come out, you have a method problem, not a result. No null should leave the team without this check.

9-11. Report the detectable effect size, not post-hoc power. "At this n and this variance, we could only have detected effects larger than D; the effect you expect is smaller than D." This is defensible, actionable, and points at the design rather than at anyone's competence. Post-hoc power calculated from the observed effect is circular and reviewers know it.

9-12. Nulls are where analytical flexibility is most dangerous. ★ The moment a dataset shows nothing is the moment the pressure to keep analyzing until something appears is highest — from the collaborator, from the calendar, and from yourself. A team that cannot report a null will manufacture false positives from its null data. There is no third option. This is the case for which blinding (Ch. 18-2) and pre-specified stopping rules were invented, and it is exactly when they will feel most inconvenient.

9-13. A null is a deliverable. An honest, well-powered negative result with a stated detectable effect size is publishable, useful, and undersupplied. An underpowered one is a design report, which is also a real product — it tells the collaborator what their next experiment must look like.
The four quadrants


Too much signal
No signal
"Prove X equals Y"
You will answer their question and miss the interesting one. Negotiate the right to publish what you find, at entry.
★ The dangerous quadrant. Maximum pressure to keep analyzing until it comes out right. Blinding and stopping rules are not optional here.
"Give me a paper"
Maximum scope risk and no stopping rule. Three years in, nothing submitted. Boundary statement and sequencing at entry.
Often the honest answer is a data descriptor, or nothing. Say so at accession, not at month eight.


[EXAMPLE — TODO] Placeholder, replace with your own: "A collaborator asked us to show that two treatment arms had equivalent immune composition; when I asked what it would mean if they did not, it emerged that the answer was already in a submitted abstract."

[EXAMPLE — TODO] Placeholder, replace with your own: "A dataset with no differential signal at all — the positive control recovered the expected sex-linked expression, so the pipeline was fine, and the honest deliverable became a detectable-effect-size statement that reshaped their next grant."
Chapter 10: Calibrating Rigor to Risk
Goal. Calibrate how much rigor a piece of work gets, rather than applying one standard everywhere. Where the wisdom comes from. Movement techniques from small-unit doctrine and margin-of-safety practice from structural engineering. What you should walk away with. Three gears, the rule never to build past your verification, and the habit of stating the margin on every conclusion.

Three gears. The error is running one gear for a whole project.

Gear
When
Practice
Fast
Consequences of being wrong are near zero
Notebook exploration, no tests, no reproducibility guarantees. Correct for the first two days.
Checked
Wrong answers would cost time but not credibility
Development with a fast test suite running behind you. You move continuously; something is watching.
Verified
Wrong answers reach other people
★ Never build further than your verification reaches. Full tests, pinned environments, review, reproducibility from scratch.


10-1. Different parts of the same project run in different gears simultaneously. The exploratory branch is fast; the release branch is verified. Teams that pick one gear either move at a crawl or ship errors.

10-2. The gear change happens at first real result, and it is a discrete event. The moment you have something you might tell someone, you are no longer exploring. Teams that keep exploring past this point never consolidate anything.

10-3. State the margin. Borrowed from structural engineering, where every design carries an explicit factor of safety over expected load. ★ Ask of every conclusion: how much would have to change for this to flip? If your result requires the effect to be exactly this size, the preprocessing to be exactly this version, and the cohort to be exactly these samples, you have no margin and you should say so out loud rather than discovering it in review.

10-4. Trace the load path. Also structural: engineers trace how force travels from application to ground, and any member that carries no load is either decoration or an error. Trace how each claim in your paper is supported down to raw data. Claims whose load path terminates in "we assumed" or "the collaborator said" are the ones that fail.

10-5. Watch tolerance stack-up. Small independent choices accumulate. Filtering, normalization, batch correction, clustering resolution, annotation, aggregation — each individually defensible, jointly capable of producing anything you like. Quantify the stack at least once per project by varying the choices jointly, not one at a time.

Public case that fits. Twenty-nine teams and sixty-one analysts were given the same dataset and the same question — whether referees issue more red cards to darker-skinned players. Effect sizes ranged from 0.89 to 2.93 in odds-ratio units; twenty teams found a significant positive effect and nine did not; the analyses used twenty-one distinct combinations of covariates. Neither prior belief nor expertise nor peer-rated analysis quality explained the spread. This is tolerance stack-up measured directly. Silberzahn et al., AMPPS 1(3):337–356 (2018); http://econweb.umd.edu/~pope/crowdsourcing_paper.pdf
Chapter 11: Problem Selection
Goal. Choose problems, which is the highest-return activity in the handbook and the one with the least process in it. Where the wisdom comes from. Hamming, and portfolio thinking. What you should walk away with. The asymmetry question, the distinction between open and load-bearing problems, and a portfolio shape rather than a queue.

The highest-return chapter in this handbook and the one with the least process in it.

11-1. Ask why you are the person who wins this. Not "is this interesting" — what is the asymmetry? A dataset nobody else has, a method nobody else can implement, a collaboration nobody else has access to, a combination of skills that is rare. A symmetric race against a larger, better-resourced group is a planning error, not bad luck.

11-2. Important problems, not merely open ones. Most open problems are open because nobody cares. Distinguish a problem that is unsolved from one that is load-bearing — where an answer would change what other people do.

11-3. Attackability is a property of the moment. The right question is not just "is this important" but "is there now an angle of attack that did not exist two years ago?" New data modality, new method, new compute, new collaborator. Importance without attackability produces a decade of frustration.

11-4. Prefer problems that generate more problems. The best projects open doors. Ask whether success would give you three new questions or zero.

11-5. Portfolio, not sequence. With two to five concurrent collaborations you already have a portfolio; manage it as one. A healthy mix is roughly: one long-horizon capability build, one or two bounded deliveries, one advisory relationship, and one thing nobody asked for. Groups drift toward all-bounded-delivery because it is the most immediately rewarded and the least cumulative.

11-6. Say no in a way that preserves the relationship. Most requests you decline should be met with something: a thirty-minute conversation, a pointer, a template, a name. This is cheap and it is what makes the yeses valuable. Declining with nothing attached costs you the relationship for a year.


PART III — PLANNING
Chapter 12: Starting a Project and Writing the Plan
Goal. Start work without serializing, and write a plan short enough to be read. Where the wisdom comes from. Troop leading procedures and the five-paragraph order. What you should walk away with. The habit of sending a warning notice before the plan is finished, and a five-section plan format that has resisted improvement for a century.

12-1. Eight steps. Receive the request → issue a preliminary notice → sketch a tentative plan → start people moving → gather information → complete the plan → brief it → supervise and refine.

12-2. Steps two and four are the point. You issue the preliminary notice and start people moving before the plan is finished. Computational teams systematically violate this by serializing — finish the design, then request the metadata, then wait six weeks for it.

12-3. The preliminary notice. Enough for others to start their own preparation. "We will need harmonized metadata in about three weeks. Exact schema TBD, but it will key on sample ID and include treatment timepoint and response." Sent today.

12-4. Charter the project on one page. Longer than one page and it will not be read; shorter and it does not constrain anything. See Appendix B.
Writing the plan
Five sections. This structure is old, borrowed, and hard to improve on.

Situation. What is known. What the data are. Who else is working this. What is published. What the collaborator believes, which may differ from what is true.
Question. One falsifiable sentence, with a purpose clause. What we are determining, and what decision or claim it enables.
Approach. Intent, concept, tasks by person, coordinating instructions, and decision points with criteria.
Resources. Compute, storage, sample budget, money, calendar time, people at what fraction.
Coordination. Who decides what. Who reports to whom, how often, on what channel. Fallback channels.

12-5. Scoping checklist. Question · obstacles and confounds · data and infrastructure landscape · people available at what fraction · calendar · stakeholders, governance, and constraints.

12-6. The plan is a hypothesis about the work. Revise it when it is wrong. Version it. A plan that never changed was never consulted.
Chapter 13: Decision Criteria and Stopping Rules
Goal. Decide in advance what would change the plan, and what would end it. Where the wisdom comes from. Aerospace flight rules — decisions argued out calmly months before they are needed. What you should walk away with. Named decision-relevant results, three kinds of gate, and the verbal go/no-go poll.

13-1. Name the results that would change the plan. Before starting, list the three to five findings that would cause you to do something different. ★ Everything else is interesting and not decision-relevant. Teams without this list chase every anomaly and finish nothing.

13-2. Write decision points with criteria, in advance. "If post-QC sample count is below N, we switch to the aggregate-primary design." "If batch structure survives correction by metric M, we drop cohort 2." Written before you look, so the decision is executed rather than negotiated.

13-3. Three kinds of gate.

Proceed criteria — what must be true to move to the next phase.
Halt criteria — what pauses the phase pending a decision.
Stop criteria — what ends the project. Chapter 25.

13-4. This is what mission rules are for. Aerospace operations write flight rules — pre-negotiated go/no-go decisions, argued out calmly months in advance, so that nobody has to make them under pressure with partial telemetry. ★ The core insight is that the same decision is made better in advance than in the moment, and the only cost of writing it down early is an afternoon. A research group's flight rules are its QC thresholds, its stopping rules, its authorship criteria, and its release standards.

13-5. Poll the room, out loud, by name. Before any irreversible step — submission, release, sending data to a collaborator, a public claim — go around the team by area of responsibility and get an explicit verbal go or no-go from each person. Silence is not assent. This takes ninety seconds and it is the highest-yield ritual in this handbook, because it creates a moment where a doubt is expected.

13-6. Record why alternatives were rejected. Not just the chosen path: the discarded approaches, the reasons, and the assumptions underlying them. ★ Six months from now someone will re-propose them. Usually you.
Chapter 14: Rehearsal and Preflight
Goal. Rehearse before it matters, and check before anything irreversible. Where the wisdom comes from. Sand-table rehearsal, simulation practice from mission operations, and preflight checklists. What you should walk away with. Figure sketches before analysis, synthetic-data dry runs, and a two-person checklist.

14-1. Sketch the figures before the analysis. ★ The highest-return fifteen minutes in project planning. Draw each figure by hand, as it would look under both the hypothesis and the null, and write the legends. If you cannot draw it, you do not have a plan; you have a hope. This also surfaces, immediately, whether the experiment can answer the question.

14-2. Dry-run on synthetic data with known ground truth. Build the pipeline against simulated data where you know the answer, before the real data arrives. You will find the bugs when finding them costs nothing. This is standard practice in physics and rare in computational biology, and the asymmetry is not justified.

14-3. Rehearse the contingency, not just the plan. Everyone rehearses the happy path. Nobody rehearses the reviewer demanding a new cohort in three weeks, or the collaborator changing the annotation two days before submission.

14-4. Preflight checklist, verified by a second person. Self-check then cross-check; the cross-check is the part teams skip and it is the part that works.

Standard preflight:

Environment locked and recorded
Seeds set and logged
Input checksums recorded
Every join asserted on row count, before and after
Output location versioned and empty
Analysis plan written and read by someone else
Stopping criteria stated
Decision owner named

14-5. Train longer than you fly. Aerospace operations run simulation campaigns where supervisors deliberately inject failures, and controllers spend far more hours in sims than in real flight. The transferable version: deliberately break your own pipeline and practice recovering. Corrupt an input. Delete an intermediate. Bump a dependency version. A team that has never rehearsed a failure will improvise its first one badly and in public.
Chapter 15: Redundancy and Fallbacks
Goal. Ensure nothing in the project has exactly one way to work. Where the wisdom comes from. Communications planning, and the redundancy/robustness/resilience distinction from structural engineering. What you should walk away with. A four-level fallback table whose empty cells are your real risk register.

15-1. Four levels for every dependency. Primary, alternate, fallback, last resort.

Domain
Primary
Alternate
Fallback
Last resort
Compute
Cluster partition
Alternate queue
Local workstation
Cloud, at cost
Data
Working copy
Institutional backup
Core's copy
Re-derive from raw
Communication
Team channel
Email
Scheduled call
Phone the person
Method
Primary model
Simpler baseline
Published alternative
Descriptive only
People
Owner
Partner
Team lead
PI


15-2. Build the table; the empty cells are your risk register. This exercise takes twenty minutes and reliably finds two or three things nobody had noticed were single points of failure.

15-3. Designate the known-good state, explicitly. The commit, the checkpoint, the version of the dataset that everyone agrees is correct — named, announced, and updated on purpose. ★ Most teams have an implicit known-good state that three people quietly disagree about, and discover the disagreement during a deadline.

15-4. Distinguish redundancy, robustness, and resilience. Structural engineering separates these and so should you. Redundancy is a second copy. Robustness is tolerating a bad input without failing. Resilience is recovering quickly after failing. They cost different amounts and solve different problems, and "we have backups" answers only the first.


PART IV — EXECUTION
Chapter 16: Standard Responses
Goal. Pre-decide the responses to problems that recur, so nobody has to reason under pressure. Where the wisdom comes from. Battle drills, and site reliability runbooks. What you should walk away with. Eight standard responses, cue-initiated, and the observation that a manual step is a defect that has not failed yet.

16-1. Why they exist. The first minutes of a problem are pre-cognitive. Under time pressure nobody reasons carefully to the right approach; they execute whatever is habitual. The purpose of a written standard response is to make the correct action the habitual one. Operations teams call these runbooks; the point is that the thinking was done in advance, by someone rested.

16-2. The standard set. One card each, rehearsed, with a named owner.

Response
Trigger
Procedure
Unexpected result
Something surprising or alarming
Reproduce with fixed seed → freeze state (commit, environment, input snapshot) → report (Ch. 33) → reduce to minimal example → owner decides: pursue, route around, or abandon
Abandoning an approach
The method is not working
Document what was tried and why it failed → commit → return to known-good state → status report → brief the team → name the alternate route
Internal defect
Bug in code you own; everything downstream suspect
Fix immediately. All results built on it are invalid until confirmed.
External defect
Bug in a dependency, scheduler, or upstream tool
Work around, pin the version, document, and move on. Do not spend three days in someone else's codebase.
External disruption
Cluster outage, collaborator delay, institutional change
Switch to the fallback (Ch. 15), notify, continue. Do not wait in place.
Pausing
Project stopping for any reason
★ State committed, environment recorded, README current, handoff note written, known-good state announced. Most teams have no procedure for stopping, which is why paused projects cannot be resumed.
Person unavailable
Illness, leave, departure
Partner assumes the task at reduced proficiency, state transfers, lead re-allocates, work is not silently dropped
Error found in released work
Published, shared, or delivered output is wrong
Immediate notification to PI, scope assessment, correction path, disclosure decision. Rehearse this one. You will need it.


16-3. Bind responses to triggers, not judgment. This QC value produces this action. Automate the trigger wherever possible; a threshold that requires someone to notice is a threshold that will not fire at 11pm.

16-4. If you have done it manually three times, automate it. Site reliability practice calls repetitive manual work with no lasting value toil, and treats eliminating it as a first-class engineering objective rather than as overhead. The corollary for research groups: manual steps in a pipeline are not a sign of care, they are a defect that has not failed yet.
Chapter 17: Tempo
Goal. Treat iteration speed as the master variable it is. Where the wisdom comes from. Boyd's decision cycle, and error budgets from site reliability engineering. What you should walk away with. Loop-count as the thing to optimize, and the discipline that fast loops are for exploration only.

17-1. Loop speed is the master variable. Run → look → interpret → decide → next run. A twelve-hour turnaround is one loop per day. A subsampled dev path is twenty. ★ The person with the faster loop explores twenty times the hypothesis space, even with a worse model. Treat "make the loop faster" as primary work, not preparation for work.

17-2. Concretely: subsampled development paths, cached intermediates, cell-based iteration over full-script reruns, a fast test that runs in under a minute, and one command that regenerates every figure.

17-3. Overlap, don't serialize. While a long job runs, someone is advancing the next question — interpretation, writing, the following model. A team where everyone waits on one job is the most common tempo failure in computational groups and it is invisible because everyone looks busy.

17-4. Fast loops are for exploration; the final analysis is slow. ★ Never let tempo justify shipping something unverified. The purpose of speed is to explore more before committing, not to commit sooner.

17-5. Budget the unreliability. Operations teams set an explicit error budget — an allowed amount of failure per period, and when it is exhausted, new feature work stops until reliability is repaired. The research version: set a threshold for reproducibility failures, broken builds, or "I can't regenerate that figure" incidents, and when it is crossed, all new analysis stops until infrastructure is fixed. Without a threshold, infrastructure work is always deferrable and is therefore always deferred.
Chapter 18: Evidence That Holds
Goal. Distinguish results that survived your analysis from results that would survive anyone's. Where the wisdom comes from. Blind analysis practice from particle physics and cosmology. What you should walk away with. Cover versus camouflage, and a set of cheap blinding techniques that almost nobody in biology uses.

18-1. Cover versus camouflage. A result that survived the analysis you chose to run looks robust. A result that survives held-out data, an orthogonal measurement, a different normalization, and a permutation null is robust. Reviewers attack the first and it fails. Know, for every figure, which one it is.

18-2. Design against your own bias, structurally. ★ This chapter's most important idea, and it is borrowed from physics rather than from operations.

Experimental particle physics and cosmology treat experimenter bias as a quantifiable engineering problem with an engineering solution, not as a character flaw to be resisted. The motivation is that even the most well-intentioned scientist is susceptible, that the scope for subtle bias is surprisingly large, and that the resulting bias represents an unquantifiable systematic uncertainty. The remedy is blind analysis: you hide the answer from yourself until the method is fixed.

Implementations vary — locking part of the data in a "black box" while the analysis is developed on the rest; "salting" the data with realistic artificial signals that are removed at the end; or adding an unknown offset to the parameters and removing it once the analysis is final. The discipline that makes it real: the collaboration agrees in advance to accept the result after unblinding *without changing any step of the analysis*, or the procedure is not blind at all. And the reason it matters: analyses tuned to maximize an observed signal produce p-values that are systematically too small and effect sizes systematically too large.

Adapted for computational biology and adjacent fields — all of these are cheap:

Develop and freeze the full pipeline on simulated or held-out data before touching the real comparison
Permute the condition labels and develop against the permuted data; unblind at the end
Hold out a subset of samples entirely until the analysis is fixed
Have one team member scramble group assignments and hold the key
Write the analysis plan and the figure legends before unblinding, and commit to publishing whatever comes out
If the pipeline was tuned after seeing the outcome, say so, or re-run on data that was not used for tuning

18-3. The point is not purity, it is that tuning-after-seeing is invisible from the outside and often invisible from the inside. A team that has never blinded anything has no way to estimate how much of its effect size is real.

18-4. Verify before release. Structural engineering proof-loads a bridge before opening it. Your equivalent: a full clean run from scratch, on a different machine, by a different person, before anything is shared.

Public case that fits. See also the Silberzahn many-analysts result noted in Chapter 10: a single dataset, a single question, and effect estimates spanning a factor of three across expert teams acting in good faith. Any single analysis you run is one draw from that distribution, and blinding is the only cheap way to stop your preferences from selecting the draw.
Chapter 19: Coordination Geometry
Goal. Prevent the damage that happens when two people converge on the same objective. Where the wisdom comes from. Fire control measures — the spatial agreements that stop friendly fire. What you should walk away with. Ownership boundaries, interface contracts, and the recognition that collaborative disasters are planning failures.

19-1. Where collaborative work actually breaks. Not skill — geometry. Two people converging on the same objective from different directions produce overlapping edits, divergent preprocessing, and two versions of "the" annotation, and the damage is discovered late.

19-2. Agree the boundaries during planning.

Ownership — who owns which module, with explicit edges
Interfaces — schemas and contracts that neither party changes unilaterally
Limits — how far a person may go without checking in
Verification before commit — never overwrite what you did not create

19-3. Most collaborative disasters are planning failures, not execution failures. Branch discipline, module ownership, and worktrees are coordination controls. Treat them with the seriousness that name implies.

19-4. Multiply by the number of collaborations. Running two to five partnerships means two to five sets of boundaries, each with its own owner, its own known-good state, and its own reporting cadence. When these blur, you deliver cohort two's annotation to collaborator one.
Chapter 20: Handoffs and Transitions
Goal. Protect the seams where work changes hands, because that is where losses cluster. Where the wisdom comes from. Clinical handoff protocols and explicit transfer-of-control practice. What you should walk away with. Assertions at every boundary, verbal ownership transfer, and handoff notes written for someone who was not there.

20-1. Losses cluster at handoffs, not during the main effort. Data leaving the core. A version bump. A cohort merge. A collaborator's re-annotation. The return from vacation. A student's last week. The bug is almost never in the model; it is in the join.

20-2. Assert at every boundary. Checksums, row counts, expected value ranges, and category levels — before and after every transfer, merge, or transformation. Assertions are cheap and they fail loudly, which is exactly what you want at 2am.

20-3. Hand off ownership explicitly and verbally. Operations practice requires the outgoing owner to state, in words, "you are now the owner of this," and to remain present until they receive firm acknowledgment. ★ Adopt this literally. Ambiguous ownership during a transition is how things fall between two people who each believed the other had it.

20-4. Use a structured handoff format. Clinical medicine has converged on standardized handoff protocols after decades of demonstrating that unstructured handoff kills people. Adapt one: situation, background, current state, what I'm worried about, what needs to happen next, who to call.

20-5. Write handoffs for someone who was not there. Including your future self, who will not remember. If the note only makes sense to a person with current context, it is not a handoff, it is a reminder.



Public case that fits. Spreadsheet software silently converts gene symbols such as SEPT2 and MARCH1 into dates, and RIKEN-style accessions into floating point. A 2016 screen of 3,597 papers found errors in about a fifth of those with supplementary Excel gene lists; a 2021 rescreen of 11,117 papers found 30.9%, and rising, despite the gene nomenclature committee having renamed the affected genes in response. The corruption enters at a handoff and propagates into every downstream reuse. Ziemann et al., Genome Biology (2016); Abeysooriya et al., PLOS Comput Biol 17(7):e1008984 (2021).
PART V — WHEN THINGS GO WRONG
Classify by what capability the response requires, not by how bad it feels. Misclassification wastes the response window.

Category
Signature
Chapter
Person unavailable or blocked
Someone cannot proceed, or is overloaded
21–22
Unexpected finding
Result is surprising, alarming, or contradictory
15
Released error
Shared or published output contains a mistake
15, 23
Approach failure
The method has failed and cannot be recovered on this axis
24
Project non-viability
The project cannot answer its question
25
Capacity exceeded
Simultaneous demand exceeds team capacity
26

Chapter 21: Triage Under Load
Goal. Impose an order on simultaneous problems so that effort is sequential rather than scattered. Where the wisdom comes from. Emergency medicine's fixed-priority assessment with interrupt-and-resume semantics. What you should walk away with. A six-step triage order, and the rule that in an active crisis you do the one thing that stops the loss and nothing else.

21-1. Fixed priority order, with interrupt-and-resume. Emergency medicine uses a fixed-sequence assessment where, when you find something that needs fixing, you pause the assessment, fix it, and resume exactly where you left off. ★ This is the discipline missing when someone debugging a pipeline notices a second problem, chases it, notices a third, and never returns to the first.

21-2. A standard order for computational teams. Adapt, then fix it and rehearse it.

Containment — is any process currently destroying data or state? Stop it first.
Preservation — is unrecoverable work at risk? Commit, copy, snapshot.
Unblocking — is anyone completely stopped? Get them moving, even imperfectly.
Recovery point — can we return to a known-good state? Establish and announce it.
Correctness — is anything already delivered or shared now known to be wrong? Assess scope.
Downstream warning — is someone else about to miss a deadline because of this? Tell them now, before you have the fix.

21-3. One thing at a time, deliberately. Stop. Look at the whole situation. Name the single highest priority. Isolate it. Work it. Confirm. Move to the next. Sequential triage feels slower than parallel effort and is dramatically faster. The hard part is the deliberate pause at the front; under stress the instinct is motion in every direction.

21-4. In an active crisis, do the one thing that stops the loss and nothing else. Do not diagnose. Do not refactor. Do not investigate the interesting adjacent thing you just noticed. Contain, then think.
Chapter 22: Escalation — Reach Versus Depth
Goal. Separate two escalations that look identical and are not. Where the wisdom comes from. Casualty evacuation doctrine, which distinguishes speed from capability. What you should walk away with. Reach versus depth, the cost of each misclassification, and the fact that depth capacity is finite and usually over-drawn.

Two different escalations. Calling the wrong one is a real and costly error, and almost nobody distinguishes them.

22-1. Reach escalation. Grab whoever is available right now, regardless of specialization, because delay is the problem. The rotating student, the person between projects, you. The goal is to move the problem out of the critical path, not to solve it well.

Call for reach when: the blocker is simple, capability is not the constraint, and the cost is the waiting. Someone needs a file moved, a job resubmitted, a meeting covered, a figure regenerated, a form signed.

22-2. Depth escalation. Route to the person with the specific capability, through a proper channel, with a structured handoff and continuity of ownership. The statistician who actually understands the mixed model. The PI who can make the call with the collaborator. The core director. Scope, context, and current state travel with the problem.

Call for depth when: the problem requires capability you do not have, and it will get worse if handled by whoever is nearest.

22-3. The two errors.

Depth when you needed reach — you wait three days for an expert's calendar on something any competent hand could have cleared in an hour.
Reach when you needed depth — the nearest available person makes a plausible-looking wrong decision, and now you have a released error instead of a blocked task.

22-4. Depth capacity is finite and must be budgeted. Your PI, your senior statistician, and your core contacts have limited throughput. Routing everything through the PI is the most common capacity failure in small groups, and it is usually invisible to the person doing it.

22-5. Send the alert before the analysis. Emergency communication doctrine is unanimous on this: the urgent request goes first with whatever information exists; the detailed report follows and never delays the request. Your full write-up follows the alert; it does not gate it.

[EXAMPLE — TODO] Placeholder, replace with your own: "I routed a blocked job submission to a senior statistician and waited four days for something a rotating student could have cleared in an hour — and in the same week handed a mixed-model question to whoever was free, which produced a confident wrong specification we then had to retract from a draft."
Chapter 23: Incident Response
Goal. Run the response when a problem reaches people outside the team. Where the wisdom comes from. Site reliability incident command and blameless postmortem practice. What you should walk away with. A named incident owner, a live timeline, scheduled updates, and the exact sentence that destroys blamelessness.

For anything that affects people outside the team — a released error, a broken shared resource, a data problem that reaches collaborators.

23-1. Name an incident owner immediately, and say the name out loud. One person coordinates. They do not have to be the most senior or the most technical; they have to be the one person everyone routes through. Without this, four people investigate the same thing and nobody talks to the collaborator.

23-2. Separate coordination from investigation. The owner coordinates and communicates; other people investigate. An owner who is also debugging stops communicating within ten minutes.

23-3. Keep a timeline as you go, not afterward. Timestamped, append-only, in a shared place. Memory of an incident is reconstructed and wrong within a day, and the timeline is what makes the review afterward worth anything.

23-4. Communicate on a schedule even when there is nothing new. "No update, still investigating, next update in an hour" is a real and necessary message. Silence during an incident is interpreted as either resolution or catastrophe, and it is never actually either.

23-5. Review afterward, blamelessly and in writing. A truly blameless review identifies contributing causes without indicting any individual or team, and assumes everyone involved acted with good intentions. The purpose is systemic improvement, not attribution.

23-6. Blamelessness is enforced by leadership behavior, not by policy. ★ Site reliability practice documents the exact failure mode: a senior person says "I know we're supposed to be blameless, but this is a safe space — someone must have known this was a bad idea, so why didn't you listen to them?" This ends blamelessness for everyone present, permanently. The recommended repair is to redirect the question generically — were there warning signs we could have heeded, and why might we have dismissed them? Practice the redirect; you will need it, and the person who needs redirecting will usually outrank you.

23-7. Review the whole response, not just the defect. Detection, mitigation, coordination, and communication are all in scope. Fixing the bug and not fixing the two hours it took to notice leaves most of the problem in place.
Chapter 24: Stopping, Pausing, and Exiting
Goal. Stop, pause, and exit as procedures rather than as silences. Where the wisdom comes from. Retrograde operations, which are planned movements rather than routs. What you should walk away with. Three distinct operations with standards, and an extraction package for a project that has collapsed.

Three distinct operations, routinely confused, all of which should be planned rather than improvised.

24-1. Abandoning an approach. This is a procedure, not a failure: document what was tried and why it failed, commit the state, return to the known-good point, file a status report, brief the team, name the alternate route. What it should not be is a dead branch and an unanswered message.

24-2. Exiting a collaboration. Requires a stated end date, a handoff package, an explicit statement of what you will and will not continue to support, and a delivered artifact even if it is only documentation. ★ An unplanned exit is read as abandonment, and that is a credibility cost you pay for years across a small field.

24-3. Holding. Minimal resource, with a stated resumption condition. Revisit on a calendar, not on a feeling.

24-4. Extraction from a collapse. When the project has genuinely fallen apart — funding ended, collaborator unreachable, data withdrawn — the question is what comes out.

Standard extraction package:

The method, generalized and separated from the dead application
The negative result, written honestly — publishable, undersupplied, and increasingly valued
The tooling, released as a package independent of the project
The people, redirected without penalty and without it appearing on their record as a failure
The review, written while it still stings
Chapter 25: Recognizing a Dead Project
Goal. Recognize when a project can no longer answer its question, and declare it. Where the wisdom comes from. The doctrinal notion of a unit becoming combat ineffective — a declared status with criteria. What you should walk away with. Five families of indicator, the three-converging rule, and pre-committed stopping criteria written before you were invested.

25-1. Non-viability is a declared status with criteria, not a mood. Declaring it is a leadership responsibility and it is not a disgrace. Failing to declare it, and continuing to feed people and money into a project that cannot answer its question, is the actual failure.

25-2. Indicators. No single one is decisive. Three or more converging means the project is not viable.

Question indicators

The question has changed three or more times, and none of the changes was driven by data
Nobody on the team can state what result would falsify the hypothesis
The decision-relevant results (13-1) can no longer be articulated, or nobody remembers setting them
The deliverable keeps shrinking while the timeline keeps extending

Execution indicators

Each new analysis exists to rescue the previous analysis rather than to answer the question
The result holds only under a specific seed, subset, parameter, or software version
Effort has shifted entirely from advancing to defending
Meetings relitigate decisions settled months ago
The last three "one more week" estimates were all wrong in the same direction

People indicators

One person carries it and everyone else has quietly disengaged
You are the only one who still believes it, and you have noticed yourself avoiding saying so
The team has stopped reporting on it and nobody has asked why
People work on it at night and it produces nothing

Partner indicators

The collaborator has stopped responding and you are constructing explanations
Their priorities have visibly shifted and they have not said so
You are the only party still investing resources

Language indicators

Sunk-cost framing dominates: "we've put two years into this"
Justification has migrated from scientific to social: "it would look bad to stop"
Discussion of the project's value has been replaced by discussion of its completion

25-3. The declaration. Made by the team lead to the PI, in writing, with the indicators listed. It is not a request for reassurance. The output is a decision: restructure, exit, or hold with a resumption condition.

25-4. Restructuring. Redistribute what remains; re-scope to a question the surviving data can actually answer; or terminate and extract per 24-4.

25-5. Pre-commit, because sunk cost is not defeated by willpower. ★ At project start, write: "We stop if, by [date], we have not [specific observable]." Put it in the charter. Review it on schedule, not when it becomes painful. The only reliable defense against sunk cost is a criterion set before you were invested.

25-6. A negative result is not a dead project. A correctly executed project that answers its question in the negative has succeeded. Non-viability is about the inability to answer, not about the answer being unwelcome. Confusing these is how groups end up unable to publish anything negative, which is both a scientific and an organizational pathology.

[EXAMPLE — TODO] Placeholder, replace with your own: "Four of the five indicators had been true for months before anyone said so out loud: the question had changed three times, every new analysis existed to rescue the previous one, one person was carrying it, and the justification had migrated from scientific to social."
Chapter 26: Capacity Exceeded
Goal. Operate when simultaneous demand exceeds what the team can deliver. Where the wisdom comes from. Mass casualty triage, where the ordinary priority rules invert. What you should walk away with. Triage by salvageability rather than severity, and the obligation to declare the state rather than absorb it silently.

26-1. Definition. Simultaneous demand exceeds what the team can deliver. Three collaborations escalate the same week; a deadline collides with a broken pipeline and a sick team member.

26-2. The rules change. Under normal load you work the most important thing first. Under exceeded capacity you work by salvageability — greatest total value, least resource. The most damaged project may be the one you deliberately do not touch this week, and that decision has to be made explicitly rather than by attrition.

26-3. Declare it. Tell the PI and the affected collaborators that you are over capacity and name what you are deprioritizing. ★ Silence during overload is read as normal operation, and everyone continues to expect normal throughput — which guarantees that you fail on all fronts instead of succeeding on some.

26-4. Some things will not be saved. Naming them explicitly, rather than spreading effort thinly across everything and losing all of it, is the entire point of triage. This is the hardest instruction in the handbook to follow and the most valuable.


PART VI — THE TEAM
Chapter 27: Composition and Coverage
Goal. Design the team so that no critical function has exactly one owner. Where the wisdom comes from. Detachment composition, where specialties are paired on the assumption of attrition. What you should walk away with. A coverage matrix, cross-training to adequacy rather than proficiency, and the hiring criterion academic groups systematically miss.

27-1. Design for absence. Small expert units in high-consequence professions deliberately pair specialties so that every critical function has at least two qualified people, on the explicit assumption that someone will be unavailable. A three-to-five person research team cannot pair everything. It can ensure that nothing has zero coverage.

27-2. Build the coverage matrix.

Function
Primary
Secondary
Gap?
Data ingest and QC






Core modeling






Software engineering and release






Statistics and inference






Domain interpretation






Infrastructure and compute






Each external collaboration






Writing and figures








27-3. Most teams find three or more functions with one owner and no backup. ★ Those are the real risks to the program — not the model architecture, which is what the group spends its meetings discussing.

27-4. Cross-train to "badly," not to "well." Every person should be able to do the adjacent job at reduced proficiency. Badly is enough to prevent a stall. The spatial person should be able to run the single-cell pipeline; the modeler should be able to regenerate the figures. Continuity, not proficiency, is the goal, and pursuing proficiency makes cross-training too expensive to happen.

27-5. Hire for team effect, not peak individual output. The strongest analyst who degrades group function is a net negative, and the arithmetic is not close. Professions that select for small-team performance screen for this explicitly. Academic hiring screens for individual output, which is why academic teams reliably acquire this problem.

27-6. Ego suppression is functional, not cultural. It is the mechanism that lets the best idea win regardless of who had it. In a room with a clinician, an engineer, and an experimentalist, the person who most needs to be right is the largest liability present, and it is very often the computational person, because they can compute things nobody else can check.

27-7. Optimal size is smaller than you think. Coordination cost grows with the square of team size. Three to five is not a limitation to be overcome; it is close to the maximum size at which everyone can hold the whole problem.
Chapter 28: Pairing
Goal. Make pairing the default rather than the exception. Where the wisdom comes from. Buddy-team practice, plus the craft tradition's account of how tacit knowledge moves. What you should walk away with. Paired ownership on anything critical, rotating roles, and the recognition that pairing is a quality mechanism as much as a redundancy one.

28-1. Pair every critical piece of work. Not review after the fact — a second person who knows the state well enough to continue it tomorrow.

28-2. No single point of knowledge in anything that has left the team. If one person's departure stops a project, that is a structural defect and it is the lead's to fix.

28-3. Alternate the roles. Fixing who runs and who reviews produces both blindness (the runner stops seeing) and bottleneck (the reviewer becomes a queue). Swap deliberately.

28-4. Check each other before irreversible steps. The preflight you run on your partner, not on yourself. Before any submission, release, or transfer, someone else runs the checklist.

28-5. Pairing is also how tacit knowledge moves. Some of what a senior person knows cannot be written down — not because they are withholding it, but because they cannot articulate it. It transmits by working alongside. This is the mechanism the craft trades rely on entirely and the one research groups use least deliberately.
Chapter 29: Crew Practice
Goal. Borrow the practice of two experts operating one system with an experience asymmetry between them. Where the wisdom comes from. Commercial aviation crew resource management. What you should walk away with. Assigned flying and monitoring roles, scripted callouts, sterile periods, and stabilized-approach gates before anything irreversible.

Commercial aviation is the closest available model for two experts operating one complex system under time pressure with an asymmetry of experience between them. Almost all of its practice transfers to pair work.

29-1. Assign the roles explicitly, every time. A flight crew designates a pilot flying and a pilot monitoring for each leg, and the roles swap. Not "we'll both watch it." One person operates; one person watches the instruments and the other person. In an analysis pair, name who is driving and who is checking, and swap by project phase.

29-2. The monitoring role is the safety-critical one, and it is the one people undervalue. Aviation learned this the expensive way. The monitor is not the junior partner or the spare — they are the error-detection system, and their attention is the primary defense. In our work the reviewer is treated as the lower-status position and it should not be. ★ Assign your most experienced person to monitor at least as often as to drive.

29-3. Call out the normal state, not only the deviation. Crews make scripted callouts at fixed points even when everything is nominal, and the stated reason is that the crew should be practiced at making the calls. A monitor who only speaks when something is wrong is a monitor who is out of practice at speaking, and whose first words carry enormous social weight. Build in routine verbalization — the standing status report, the go/no-go poll (Ch. 14-5) — so the channel is warm before it is needed.

29-4. Sterile cockpit. Below ten thousand feet, non-essential conversation is prohibited. The rule exists because the phases with the highest workload are also the phases most vulnerable to interruption, and because "just one quick question" is a documented accident precursor. ★ Designate sterile periods for your team: the day before a submission, the hours during a live data transfer, the final figure assembly. No Slack, no drop-ins, no unrelated meetings. Say it out loud so people know it is a rule and not a mood.

29-5. Stabilized approach criteria. This is the strongest single import in the chapter. An approach is stabilized when a defined list of conditions is met — correct flight path, correct speed within a stated tolerance, correct configuration, descent rate below a threshold. Crews evaluate these at fixed gates, and modern practice uses a "should" gate followed lower down by a "must" gate. If the aircraft is not stabilized at the must gate, a go-around is mandatory regardless of how it looks out the window.

Adapted to a manuscript or release, define your gates as a checklist evaluated at fixed points before submission — one month, one week, one day — with named criteria: every figure regenerable from committed code, every number in the text traceable to an output, no analysis performed after the plan was frozen without documentation, all authorship agreed, all data deposited. At the must gate, if the criteria are not met, the submission slips. Not "we'll fix it in proof."

29-6. The go-around is free, and everyone still resists it. Aviation's honest finding is that compliance with stabilized-approach policy is lower than expected, because pilots do not believe the approach is beyond saving at that altitude — they think they can rescue it, and often they land without consequence, which reinforces the habit. The counter-practice: the monitor calls out the deviation continuously as the gate approaches, so that the decision is made by an agreed rule rather than by the optimism of the person flying. ★ Note the mechanism carefully: the person who calls the go-around is not the person who wants to save it. Give that authority to the monitor.

29-7. Aviate, navigate, communicate. A fixed priority ordering for overload: fly the aircraft first, then figure out where you are, then talk to anyone. Adapted: keep the system running, then establish where you actually are, then report. The failure mode it prevents is the crew that talks to the tower while flying into the ground — or the analyst who writes a long explanatory message while the job that is corrupting the output continues to run.

29-8. Threat and error management. The modern framework: identify threats (conditions that increase risk and that you did not cause — a new collaborator, an unfamiliar assay, a compressed deadline, an unmaintained dependency), trap errors before they propagate, and recognize undesired states early enough to recover. The reframing that matters is that threats are expected and briefed in advance, not treated as bad luck when they arrive. Add a threat line to every project brief: what conditions in this project make error more likely?

29-9. Two kinds of checklist, and they are not interchangeable. A read-do checklist is followed step by step during an unfamiliar or infrequent procedure. A challenge-response checklist verifies, afterward, that steps already performed from memory were in fact performed — one person challenges, the other responds. ★ Most scientific checklists are written as read-do and used as challenge-response, badly. Decide which each of yours is. Preflight (Ch. 15-4) is challenge-response and requires two people. A rarely-executed release procedure is read-do.

29-10. Positive transfer of control. Handover of the controls is verbal, explicit, and confirmed: one pilot states they have the aircraft, the other confirms they have relinquished it. There is never a moment where both or neither believe they are flying. Chapter 20-3 says the same thing about task ownership; aviation is where the practice was proven, and the reason it is scripted rather than assumed is that the ambiguous handover is the one that kills people.
Chapter 30: Working Through Others
Goal. Work effectively where you have responsibility and no authority. Where the wisdom comes from. Foreign internal defense doctrine, whose entire premise is operating through partners. What you should walk away with. Success measured as the partner's self-sufficiency, support intent specified at the outset, and exits designed at entry.

This chapter governs the majority of your operational time.

30-1. You have no authority. You cannot direct a clinician to standardize metadata or an experimentalist to change a protocol. Your levers are credibility, competence, relationship, and making the correct path the easy path.

30-2. The measure of success is their self-sufficiency. A collaboration where they cannot function without you has failed, however many papers it produced — it has consumed your capacity permanently and left them dependent.

30-3. Deliver something they value early and small. Credibility comes from shared experience and there are no shortcuts. The first deliverable should be fast and useful to them, not the most scientifically interesting thing available to you.

30-4. Do not build what they cannot maintain. A pipeline requiring your cluster, your environment, and your presence is a dependency you created and will service indefinitely.

30-5. Each partnership is a separate environment. Different skill mix, funding, priorities, tempo, and definition of success. For each collaboration write down: their goal, their binding constraint, their decision maker, their timeline, and what "good" means to them. These will not match each other and will not match yours, and assuming otherwise is the source of most collaboration friction.

30-6. Specify the intent of your support. From 7-7. Unspecified support obligations expand until they consume the team.

30-7. Agree the exit at the start. From 7-4. What you deliver, when you leave, what you hand over, what you will not maintain.

30-8. Share early and by default, within agreed rules. Large-scale collaborative science converged on this the hard way. The Human Genome Project's Bermuda Principles, adopted in 1996, required that sequence assemblies above a certain size be submitted to a public database within twenty-four hours of generation. The practice originated in the C. elegans community, and in both settings daily sharing served the pragmatic purposes of quality control and project coordination — not idealism. Fast internal sharing catches errors early and keeps parallel groups synchronized, and that is the argument to make to a reluctant collaborator, rather than the ethical one.

30-9. Write the data agreement before the data moves. Who may use it, for what, with what embargo, with what authorship expectation, and who resolves disputes. Ten minutes of awkwardness at the start prevents the most common and most damaging category of collaboration failure. Note also the documented pathology: conditioning data access on co-authorship is a widely recognized obstacle to reproducibility, and you should be careful not to become the person doing it.

30-10. Authorship criteria are decided at project start, in writing. Not at submission. Everyone involved should be able to state, in advance, what would earn them what position. This one page prevents more damage than any other page in the project.
Chapter 31: Growing People
Goal. Develop people deliberately, since nothing else in the institution will. Where the wisdom comes from. Craft apprenticeship. What you should walk away with. The five-step ladder, practice pieces with no deliverable, and the obligation to let juniors watch you choose rather than only execute.

31-1. Delegation is bounded by demonstrated competence — which is a reason to build competence deliberately, not a reason to withhold work.

31-2. Support the outcome of decisions you delegated. Including the wrong ones, when they were made in good faith on the information available. This is the entire mechanism by which initiative survives.

31-3. The apprenticeship ladder. Craft traditions use a sequence that research groups apply haphazardly: watch → assist → do under observation → do alone → teach. Skipping steps produces people who can execute but cannot judge. The final step is not a courtesy; teaching is where understanding consolidates, and a group that never asks juniors to teach is leaving that consolidation on the table.

31-4. Practice pieces. Craftspeople make things they intend to discard, purely to build skill. The scientific equivalents — reimplementing a published method from scratch, analyzing a dataset with a known answer, writing a review nobody asked for — are enormously effective and almost never assigned, because they have no deliverable.

31-7. Let people watch you choose. Juniors mostly see seniors execute. What they need to see is the selection — why this problem and not that one, why this method, why we are stopping. Narrate your reasoning out loud when you make these calls. This is how taste transmits, and there is no other mechanism.
Chapter 32: The Conditions for Honesty
Goal. Create the conditions under which people say inconvenient things out loud. Where the wisdom comes from. Manufacturing quality practice, aviation assertiveness training, and high-reliability organization research. What you should walk away with. Stop-the-line authority, a graduated assertiveness script, and the arithmetic that makes a false alarm cheap.

Everything in Parts IV and V depends on people saying inconvenient things out loud. That is a property of the environment, not of individual courage.

32-1. Anyone may stop the line. ★ The most junior person must have explicit, repeatedly stated authority to halt a submission, release, or claim over a suspected problem — without needing to be sure. This is a direct import from manufacturing quality practice, where any worker may stop production, and it is the single highest-value cultural rule in this handbook. Its absence is how known-wrong things get published.

32-2. Say it out loud, repeatedly, and then demonstrate it. The rule means nothing until someone junior uses it and is visibly thanked rather than visibly tolerated. Engineer that first instance if you have to.

32-3. Graduated assertiveness. Aviation crew training teaches an escalation ladder so that a junior crew member has a script for disagreeing with a captain: state the observation, state the concern, propose an alternative, and then explicitly challenge. Teach the same ladder. "I notice the batch variable isn't in the model" → "I'm concerned this could be a batch effect" → "Could we run it with batch included?" → "I don't think we should send this until we've checked." People who are given the words use them; people who are only given permission usually do not.

32-4. Watch for language that closes the room. Not just blame — dismissal, weary sighs, "we've been through this," and the phrase "I'm sure it's fine." Each of these costs you one future warning.

32-5. Reward the false alarm. If someone raises a concern that turns out to be nothing, the response must be visible appreciation. ★ The cost of a false alarm is fifteen minutes. The cost of the suppressed true alarm is a retraction. Price them accordingly and say so.

32-6. Never accept success as evidence of rigor. Aerospace mission operations names vigilance as a core value, defined as never allowing success to substitute for rigor in anything. A project that worked out does not mean the process was sound; it means the process was not tested this time.



Public case that fits. The MD Anderson group that found the errors in the Duke chemosensitivity work reported that their correspondence was rebutted, that letters to cancer journals identifying further errors were rejected, and that their results were treated as too negative to publish; they eventually published in a statistics journal, and the trials continued enrolling patients in the interim. The failure was not detection. It was that the institution had no channel through which an inconvenient finding could travel.
PART VII — COMMUNICATION
Chapter 33: Report Formats
Goal. Give reports a fixed shape so the receiver can tell what is missing. Where the wisdom comes from. Military reporting formats, which exist because unstructured reports fail under load. What you should walk away with. Three formats, the rule that the alert precedes the analysis, and the mandatory negative report.

33-1. Why fixed formats. Compression works only if the receiver already knows what is coming. Fixed field order lets a listener allocate attention correctly and — more importantly — notice what is missing.

33-2. The finding report. For anything surprising, and it goes out fast rather than polished.

Magnitude — effect size, count, fold change
What — the observation, concretely
Where — which cohort, sample, module, branch
Provenance — code version, environment, seed
When — observed, and introduced if known
How measured — the metric and its assumptions
Actions taken and recommendation

33-3. The escalation request. Most urgent first, fixed order, sent partial rather than complete.

Where the problem is
How to reach me, on what channel
Severity and category (Ch. 23)
What capability is required — this determines reach versus depth (Ch. 22)
What is blocked, and who is downstream
What has already been tried
The deadline
What decision I need from you
What I will do if I do not hear back

33-4. The status report. After every milestone: resources remaining (compute, time, budget, calendar), losses (what broke, which assumptions died, what was dropped in QC), assets (what code and intermediates now exist and are reusable).

33-5. Negative reports are mandatory. ★ "No progress this week, here is why" is a real and required report. Silence is read as either fine or catastrophic and it is never actually either. This is the cheapest high-value habit in the handbook and the most commonly skipped.
Chapter 34: The Shared Vocabulary Problem
Goal. Solve the problem that cross-disciplinary teams share words without sharing meanings. Where the wisdom comes from. Brevity codes, and closed-loop communication from aviation and surgery. What you should walk away with. A project glossary in week one, a fixed shape for reporting results, and readback on anything consequential.

34-1. Compression requires a shared decode key, and across disciplines there is none. Worse, there are false friends: words that both parties use confidently, in the same meeting, meaning different things.

34-2. The standard false friends. Significant. Expression. Model. Validated. Cluster. Progression. Response. Control. Replicate. Batch. Power. Confidence. Bias. Sample. Normalization. Each carries a different meaning to a statistician, a clinician, an experimentalist, and an engineer.

34-3. Build the glossary and data dictionary in week one. ★ The highest-yield communication intervention available in cross-disciplinary work, and almost nobody does it. Two pages. Maintained. Referenced in meetings when a term is used ambiguously.

34-4. Report results in a fixed shape. What changed, how much, in what population, with what confidence. "CD8 fraction up 2.3-fold in responders, n=14 versus 11, holds after correcting for within-patient correlation" is a report. "We found interesting immune differences" is noise wearing a lab coat.

34-5. Read back critical information. "To confirm: responders are defined by RECIST at six months, not best overall response." Closed-loop communication is standard in aviation, surgery, and mission operations for the same reason: the sender's confidence that they were understood is uncorrelated with whether they were. Do this in every clinical meeting, every time.

34-6. Attention is the scarce channel. Meeting time and collaborator bandwidth cost the most from the people whose time is hardest to get. Chatter is not free.

34-7. Know the fallback channel in advance. For each collaborator, how do you reach them when the primary channel fails? It will fail at the worst moment, and the fallback should not be improvised then.


Chapter 35: Scientific Writing
Goal. Treat writing as the craft it is rather than as reporting. Where the wisdom comes from. Woodworking's sequence of operations, reader-expectation theory, and the empirical literature on writing habits. What you should walk away with. Structure before prose, four separable passes, four mechanical sentence rules, and short daily sessions.

Chapter 2-12 asserted that writing is thinking rather than reporting. This chapter is the craft.
Structure before prose
35-1. Rough, dimension, joint, finish — in that order. A woodworker does not finish a surface before the piece is dimensioned, because finishing an oversized board is wasted work. ★ The most common failure in scientific writing is polishing sentences inside a structure that is wrong. The four passes are separable and must not be mixed: structure (does the argument hold), paragraph (does each unit do one job), sentence (is each readable), finish (typography, references, consistency). Editing a sentence you will later delete is the most reliable way to feel productive while accomplishing nothing.

35-2. Write the outline before the work is done. Whitesides' method: the outline is not a summary of completed research, it is a research plan — the figures you intend to have and the claims they will support, written early, revised continuously. It tells you which experiments matter and which are decoration. This is the same instrument as the figure sketch in Chapter 13-1, extended to the argument.

35-3. One central contribution. A paper carries one idea. Everything else is support or it is a different paper (Ch. 9-6). ★ The test: state the contribution in a single sentence containing no conjunctions. If you need "and," you have two papers or an unfinished thought.

35-4. Context, content, conclusion — at every scale. The same three-part shape governs the paper, each section, each paragraph, and often each figure. Fractal structure is what makes a long document navigable, because a reader who understands the pattern can enter at any level and orient.

35-5. The load path applies to prose. Chapter 10-4 asked you to trace how each claim is supported down to raw data. Do it in the manuscript explicitly: every claim in the abstract traces to a claim in the results, which traces to a figure, which traces to committed code and to data. ★ Any sentence whose load path terminates in "it is well known" or "we assumed" is either decoration or a defect, and reviewers find both.

35-6. Kill the expensive things first. The analysis that took six months is not thereby load-bearing. The strongest signal that a section should be cut is usually that you are reluctant to cut it, and the reluctance tracks cost rather than contribution. Move it to supplementary, keep it for the next paper, or delete it.
Sentences, mechanically
35-7. Reader-expectation theory is the one piece of writing advice that is mechanical rather than aesthetic. Gopen and Swan's argument (American Scientist, 1990) is that readers derive meaning not only from words but from where in a sentence the words appear, and that these positional expectations are consistent enough to be exploited deliberately. Four rules cover most of it:

Topic position — the beginning of a sentence tells the reader what this sentence is about. Put the thing you are discussing there, not a subordinate clause of throat-clearing.
Stress position — the end of a sentence is where emphasis lands. Put the new, important information there. A sentence that ends on a citation or a caveat has thrown away its most valuable real estate.
Old to new — begin with what the reader already knows, end with what is new. This is what makes a paragraph feel like it flows; flow is not a mystery, it is topic-and-stress chained across sentences.
Keep subject and verb close. Everything between them is held in working memory at cost, and scientific prose routinely inserts thirty words there.

★ These are teachable in an afternoon and they improve a junior's writing more than any amount of "be clearer." Teach them explicitly.

35-8. The curse of knowledge is the root cause of most bad scientific prose. Not ego, not obfuscation — the writer cannot reconstruct not knowing the thing. The countermeasure is structural rather than motivational: a reader who does not have your context, reading before submission, instructed to mark every place they had to reread. Do not ask them whether it was clear; ask them where they stopped.

35-9. Clutter. Prefer the shorter word, the active voice where the actor matters, and one clause where two will do. But do not confuse brevity with clarity: an unclear short sentence is worse than a clear long one, and scientific writing has a genuine need for precision that general style advice underweights.
Practice
35-10. Brief daily sessions beat writing binges, and this is an empirical finding rather than a preference. Boice's work on academic writers found that people who wrote in short scheduled sessions produced substantially more and reported less resistance than those who waited for blocks of free time — and that the binge writers generated fewer new ideas as well. ★ Thirty to sixty minutes, scheduled, most days, before anything else. The single largest lever available in this chapter.

35-11. Writing is the audit of the work. Chapter 2-12: the gaps in the prose are the gaps in the analysis, and they surface far earlier in writing than in code. Start writing in week two. When you cannot write the paragraph, stop and find out whether you have the result — that hesitation is diagnostic information, not a writing problem.

35-12. The abstract is a contract, so draft it first. Write it before the analysis is finished, as a statement of what you intend to be able to claim. It will change. Its value is that it forces the contribution into one paragraph while the design is still adjustable, and it becomes the reference against which scope creep is visible.
Team writing
35-13. One pen. A paper drafted by committee reads like one. Assign a single author who owns the prose; everyone else comments. Sections drafted in parallel by different people must still be rewritten by one voice, and that rewrite is a real task with real hours, not a formality.

35-14. Comment on the level currently in play. Chapter 35-1's passes apply to review as well: sentence-level edits on a draft whose structure is still contested waste everyone's time and, worse, they signal that the structure is settled. State which pass you are reviewing at. "Structural comments only" on the covering message.

35-15. Response to reviewers is its own genre and it is taught nowhere. Its properties: answer every point, in order, visibly; separate what you changed from what you are declining and why; quote the reviewer before responding; concede readily where the point is fair, because conceded points buy credibility for the ones you contest; and never argue with tone. ★ Write the response, then wait a day, then remove every sentence that is defending rather than answering.

35-16. Writing is where the junior members learn what the work was for. Do not have the lead write the paper because it is faster. Chapter 46-6 applies: the first draft is where compilation happens.
Chapter 36: Channels and Cadence
Goal. Operate across channels you do not control, without losing the record. Where the wisdom comes from. Aviation's guard frequency, ship's log practice, and influence-without-authority mechanics. What you should walk away with. Meet them on their channel and log it on yours, published response times, and the observation that whoever writes the record shapes it.

You will run several concurrent collaborations, each with its own established habits, and you will have no authority to change any of them. Your PI texts. One collaborator only takes phone calls. The same question arrives on Slack, email, and Teams. This chapter is about operating well inside that, rather than fixing it.

36-1. Do not fight the channel. Fight the loss of the record. ★ The problem is not that people talk in different places; it is that decisions get made in ephemeral channels and then cannot be found. You cannot control where a conversation happens. You can control where its outcome lives, and that single separation resolves most of the chaos without requiring anyone to change anything.

36-2. Meet them on their channel; log it on yours. The phone collaborator gets a phone call, followed by a summary email. The PI who texts gets a text reply, followed by a line in the project document. The Teams thread gets a Teams answer, followed by the decision recorded where your team keeps decisions. One extra step per interaction, and it removes the entire class of "where did we land on that."

36-3. Channels have properties, and messages have requirements. Match on the property, not on habit:

Property
Matters for
Synchronous / async
Anything requiring negotiation or disagreement should be synchronous. Anything requiring precision should be written.
Durable / ephemeral
Decisions, commitments, and numbers must be durable. Coordination need not be.
Searchable
If someone will need it in six months, it must be findable by someone who does not know it exists.
Discoverable by others
Private channels create knowledge that only two people have — Chapter 28-2's single point of knowledge, created accidentally.
Attributable
Anything that assigns work or ownership needs a name on it.


★ The common error is sending a decision through an ephemeral, unsearchable, private channel, which is exactly what a text message is.

36-4. Publish response-time expectations per channel. This is the intervention that actually stops cross-posting, because cross-posting is an anxiety behaviour: people duplicate because they do not know when they will hear back. State it once, in a pinned message and your email signature: "Slack — same working day. Email — two working days. Text or phone — genuinely urgent." People comply with clear expectations far more readily than with requests to change tools.

36-5. One guard channel. Aviation reserves a single emergency frequency that everyone monitors regardless of what else they are doing. Have one channel that means this is urgent and protect it ruthlessly by never using it for anything else. The moment routine traffic appears on it, it stops working, permanently.

36-6. Fixed cadence reduces ad hoc interruption more than any policy. A predictable weekly written update per collaboration eliminates most "just checking in" traffic, because the checking-in is anxiety about not knowing. Chapter 33-5's negative report is the instrument: it goes out on schedule whether or not anything happened, which is what makes it trustworthy.

36-7. Summarize and confirm, always. "Summarizing what we agreed: A, B, C. Let me know if I have that wrong." Chapter 34-5's readback, extended to every substantive conversation regardless of channel. It costs two minutes, it catches genuine misunderstandings at a rate that will surprise you, and it produces the record in 39-1.

36-8. The person who writes the record shapes the record. ★ This is the honest answer to how you exert influence without authority, and it is not manipulation — it is a real service that happens to confer real standing. The summary you write becomes what everyone remembers. The person who reliably produces the clear, fair, prompt account of what was decided becomes, over a year or two, the person consulted before decisions are made. Volunteer for this. It is the cheapest available route to influence in any organization and almost nobody takes it, because it looks like secretarial work.
Influence without authority, concretely
36-9. Never ask people to change tools. Change what is in the tool. You cannot make colleagues adopt Slack by advocating for Slack. You can make it the place where the answer already is. ★ If searching your Slack reliably answers a question faster than asking a person, adoption follows without anyone deciding to adopt. This is a content problem, not a policy problem, and it is entirely within your control.

36-10. Reduce their effort, never increase it. Every proposed norm should be net-negative work for the person adopting it, in the first week, or it will not take. A template that saves them writing. A dashboard that answers the question they were going to ask. A summary they did not have to produce. Norms that cost effort up front for collective benefit later require authority; norms that save effort immediately require none.

36-11. Model it silently, name it once. Do the thing consistently for three months before mentioning it. Then say it exactly once, as a description rather than a request: "I keep decisions in this doc so I can find them; you are welcome to use it." Repeated advocacy converts a useful practice into your personal campaign, and people resist campaigns.

36-12. Make artifacts, not requests. A request is a debt the other person owes you. An artifact is a gift they can use. ★ The staff scientist's whole influence model reduces to this: produce things so useful that adopting your way of working is the path of least resistance. Chapter 30 says the same thing about collaborators; it is equally true inside your own institution.
PART VIII — TOOLS AND ENVIRONMENT
Chapter 37: The Loadout
Goal. Decide what you carry, and make sure the lowest tier survives everything else failing. Where the wisdom comes from. Load-out doctrine, small-unit weapons selection, field engineering, and seamanship. What you should walk away with. Three tiers, an explicit command-line inventory, the figure last mile, and tool selection by half-life.

Every profession that works away from its base has a theory of what you carry. The theory is always the same shape: tiers, each independently functional, each higher tier discardable, and the lowest tier chosen so that losing everything else is survivable rather than fatal.

37-1. Three tiers.

Tier 1 — on the body. The shell, version control, the language interpreter, the ability to read a stack trace, and the statistics in your head. Available on any machine, with no configuration, over a bad connection.
Tier 2 — the working kit. Project template, analysis library, environment manager, standard plotting module, QC script, preflight checklist, the editor you are fast in. Sustains a working session.
Tier 3 — the heavy load. Cluster jobs, large intermediates, full raw data, orchestration, the GUI tools. Powerful, immobile, and discardable by design.

37-2. The test for Tier 1. ★ Dropped onto a bare account on an unfamiliar machine, with no dotfiles, no IDE, and no web portal — can you still do your job? If productivity collapses without a specific editor configuration, Tier 1 is too thin, and the day the portal is down is the day you find out.

37-3. What Tier 1 actually contains. Make this explicit and make everyone proficient. Not expert — proficient, meaning they can do it under pressure without searching.

ssh, including key management, config files, jump hosts, and agent forwarding
Port forwarding and tunnels, in both directions. Launch a notebook or server on a compute node and reach it from your laptop, through a login node, without the portal.
A persistent session multiplexer — tmux or screen — because your connection will drop mid-job
Scheduler CLI: submit, query, cancel, inspect resource usage, read the accounting logs, figure out why a job was killed
File movement: rsync and scp, including resumable transfers and checksum verification
A terminal editor, at minimum survival-level vim or nano
Text processing: grep, sed, awk, sort, uniq, cut, head, tail, and pipes — enough to inspect a 40 GB file you cannot open
git from the command line, including recovering from a bad state
Process and resource inspection: top/htop, ps, du, df, lsof, and knowing what to do when a filesystem is full
Environment management from the shell, and the ability to build one from scratch when the shared one breaks
tar, gzip, checksums, and permissions

37-4. Why this is not nostalgia. ★ A GUI is a single point of failure that sits between you and everything you own. Web portals go down, licenses lapse, IDEs break on upgrade, and remote-development extensions fail exactly when the cluster is under load. Every one of those is recoverable in minutes by someone with Tier 1 and is a lost day for someone without it. It is the same principle as Chapter 38-5: your recovery path must not depend on the thing that broke. The portal is a convenience layer over ssh and a scheduler. Convenience layers are correct to use and fatal to require.

There is a second reason, less often stated. Command-line fluency changes what you attempt. A person who can pipe, filter, and inspect a file in place will look at raw data directly; a person who cannot will only ever see data that has already passed through something. That difference compounds into scientific judgment.

37-5. The last mile, which is where people get burned. ★ Figure generation is the one place where the final ten percent is done in a different tool than the first ninety, and it is the place where the team is most likely to be blocked at the worst moment.

The discipline has two parts. First, protect the option: export vector formats — SVG or PDF — not rasters, so that final assembly is possible at all. A 300-dpi PNG at deadline is an unrecoverable position. Second, be able to do the assembly yourself: panel arrangement, alignment, typography, colour correction, and label editing, in Illustrator, Inkscape, or Affinity. Handing the last mile to someone who does not understand the science is how axis labels end up wrong in print, and it is a dependency that activates precisely during the final push, when nobody has slack.

The corollary is a Tier 2 obligation: your plotting code should get the figure to about ninety-five percent, so that the manual step is arrangement and polish rather than reconstruction. Anything you touch by hand more than twice belongs in the code.

37-6. Do you use every available tool? No. Three professions converge on the same answer and it is not the intuitive one.

The painter's answer is the limited palette: constraining yourself to a few pigments forces you to learn what they actually do in combination, and produces more coherent work than a hundred tubes. The craftsman's answer is that tool count is inversely correlated with skill in mature practitioners — a beginner buys tools, an expert owns few and maintains them. The operator's answer is that every item is carried on every deployment, so a tool must earn its weight against the endurance it costs.

The common mechanism: depth of familiarity beats breadth of availability, because the returns to knowing a tool completely are nonlinear and the returns to owning one more are nearly zero.

37-7. Criteria for adopting a tool. Adopt when it clears all four:

Does it remove a category of error, or a category of tedium? Anything else is a preference.
Does it survive the borrowed-machine test? If it cannot be installed in five minutes on a fresh account, it is Tier 3 at best.
Is it maintained, and by whom? A tool with one unpaid maintainer is a dependency with a known end date.
Would you still choose it if you had to teach it to a new team member next month? This one eliminates most of what survives the first three.

37-8. Invest by half-life. ★ Tools have wildly different decay rates and most people invest as though they do not:

Decade-scale: the Unix shell and its text utilities, ssh, git, SQL, a general-purpose language, vector graphics, regular expressions, the mathematics. Anything learned here is still paying out in fifteen years.
Few-year scale: the current frameworks, wrappers, orchestration layers, notebook platforms, and IDE ecosystems. Useful, worth using, and not worth deep investment.

Spend learning time asymmetrically toward the first list. Adopt from the second list freely and hold it loosely.

37-9. Team loadout versus individual loadout. The tiers are governed differently, and the intuition is backwards:

Tier 1 must be universal. Everyone can operate bare. This is non-negotiable and it is a training obligation, not a preference. It is the team's floor under any infrastructure failure.
Tier 2 must be standardized. Shared project scaffolding, conventions, and libraries, because this is the layer at which one person's work has to be legible and runnable by another (Ch. 47-6).
Tier 3 is personal. Editor, theme, keybindings, shell configuration, local tooling. Let people be idiosyncratic here; it costs the team nothing and it is where the pleasure is.

Getting this backwards — standardizing the editor while leaving project structure to taste — is common and exactly wrong.

37-10. The cold-machine drill. Once or twice a year, on a fresh account with no configuration: clone the repository, build the environment, submit a job, tunnel into a notebook running on a compute node, regenerate a published figure. Time it. ★ This is the equipment check for the whole chapter, it finds the Tier 1 gaps that nobody will admit to, and it is the single best onboarding exercise for a new team member.

37-11. Commonality of resupply, from small-unit weapons doctrine. Detachments operating alongside partner forces value weapons that share ammunition, magazines, and spare parts with everyone around them, because a superior weapon you cannot resupply is worse than an adequate one you can. ★ Prefer the tool your collaborators, your core, and your field already use. An exotic framework that is better on the merits but that nobody else can help with, hire for, or take over is a permanent dependency on you personally. Resupply, in our terms, is help, hiring, handoff, and Stack Overflow.

37-12. One platform, many configurations. Modular weapon systems keep a single common core and swap barrels, optics, and furniture by mission, so that the manual of arms never changes. ★ Carry one core environment and one primary language, and configure per project. A team split across three ecosystems pays the switching cost on every handoff, every code review, and every hire.

37-13. Zero your own equipment. A rifle is useless until it has been zeroed to the individual and confirmed by shooting it. ★ A shared configuration nobody has personally verified is a liability, not a resource. When you adopt the team template, run it end to end yourself once before you rely on it.

37-14. Never carry what you have not trained with. The deadline project is not where you first use a new library, a new scheduler, or a new plotting system. ★ New tools are adopted in the fast gear (Ch. 10), on work that does not matter, and promoted only after they have survived something real.

37-15. Reliability beats peak performance. The recurring preference across every operational profession is the thing that works every time over the thing that works better when it works. ★ A method that is 5% better and fails on a quarter of your datasets is worse than the boring one, and this is the trade computational scientists most consistently get wrong, because the 5% is publishable and the failure rate is invisible until it is not.

37-16. Redundancy of function, not duplication of item. A sidearm is a backup because it fails differently from a rifle, not because it is a second rifle. ★ Your backup should differ in mechanism: a simple baseline model against a complex one, a hand calculation against a pipeline, an orthogonal assay against the primary. Two implementations of the same approach share their blind spot and give you false confidence.

37-17. The field expedient, and marking it. Field engineering distinguishes the expedient repair — done with what is on hand, correct enough to continue the mission — from the permanent one, and the discipline is that the expedient is marked so that it gets replaced. ★ The hack is legitimate. The silent hack is not. Every workaround carries a named, dated marker in the code and a line in a single visible list, or it becomes permanent by default and nobody will ever know it was provisional.

37-18. Standard designs, assembled under bad conditions. Military bridging systems are built from standardized, interchangeable parts so that a structure can be assembled correctly by tired non-specialists at night. ★ This is the design target for your internal modules: not maximal capability, but correct assembly by a tired person who did not write them. If a component requires its author present, it is not a standard design.

37-19. Reef early — the sailor's rule. The time to reduce sail is the moment you first think about it, not when the conditions force it, because by then the job is harder and more dangerous. ★ The time to cut scope is the moment you first wonder whether you should. This is the most transferable single sentence in seamanship and it applies to project scope, author lists, analysis breadth, and the number of concurrent engagements.

37-20. One hand for yourself, one hand for the ship. Sailors are taught to maintain their own hold while working, because a crew member who is lost helping is a net loss to the vessel. ★ Applied twice: keep one hand on your own skills and career while serving the group's projects, and do not let a team member spend themselves entirely on a project that will not remember them.

37-21. Standing rigging and running rigging. Standing rigging holds the mast up and is adjusted rarely, under controlled conditions, with great care. Running rigging is handled constantly. ★ Know which of your systems is which. Environment definitions, data schemas, storage layout, and the release pipeline are standing rigging — changed deliberately, never mid-passage. Analysis code is running rigging. Teams that treat standing rigging casually lose the mast.

37-22. Maintain dead reckoning even with a satellite fix. Navigators keep an independent estimate of position by course, speed, and elapsed time, and do not abandon it merely because an instrument is reporting a position. ★ Maintain your own analytical estimate of what the answer should be, independent of the model's output. When the two diverge, that is the most informative moment available to you, and you only get it if you kept the estimate.
Chapter 38: Weight Discipline
Goal. Account for the cost of everything you carry. Where the wisdom comes from. Load-weight research, and the workshop habit of converging by subtraction. What you should walk away with. The permanent cost of a dependency, consistency over cleverness, and the diagnostic that distinguishes tool acquisition from capability.

38-1. Every dependency is carried forever. On every install, every import, every onboarding, every debugging session, for the life of the project. The framework that saved twenty lines and costs a debugging session in year two was a bad trade, and the trade is usually made without being noticed.

38-2. Abstractions far from your actual problem cost the most. A domain-specific helper is cheap. A general framework you use one feature of is expensive.

38-3. Consistency over cleverness. Same structure, same naming, same import block, same figure-output convention across every repository. Under pressure you index by habit, not by looking.

38-4. Nothing loose. Pin versions. Lock environments. Fix seeds. Shake the whole rig before depending on it — a clean end-to-end run from scratch, on a machine that has never seen the project.

38-5. Recovery must not depend on the thing that broke. Backups on the machine that failed are not backups. Documentation only in the head of the person on leave is not documentation.

38-6. Redundancy on the three things that actually matter. Raw data integrity, provenance and metadata, and the ability to regenerate a published figure. Everything else is recoverable.

38-7. Converge by subtraction. Nobody designs a good working setup on paper. Log what you actually reach for over six months and delete the rest. Your utilities module contains functions written once and never called again; you carry them on every import.
The acquisition trap
38-8. There is a well-made critique in operational communities that the visible, purchasable markers of professionalism have started to displace the actual competencies — that people invest heavily in better-looking equipment while subconsciously compensating for professional shortcomings. It applies to us with almost no translation.

38-9. Our version. The editor configuration. The terminal theme. The font. The dotfiles repository. The new GPU. The eleventh workflow tool. Tool acquisition feels exactly like capability development, produces a visible artifact, activates the same reward circuitry, and is measurably easier than the thing it displaces.

38-10. The diagnostic. When did you last change your tooling, and when did you last change your method? If the ratio is worse than one to one, you are optimizing the wrong layer.

38-11. This is not an argument for bad tools. Good tools compound and Chapter 39 is entirely about building them. It is an argument for noticing which activity you are actually doing.


Chapter 39: Jigs, Fixtures, and Sharpening
Goal. Justify building the thing that makes the work correct. Where the wisdom comes from. The workshop — jigs, fixtures, sharpening, and mise en place. What you should walk away with. Build the fixture before you need it forty times, and identify which operations are irreversible so care can be concentrated there.

Borrowed from the workshop, and the most underused idea in computational practice.

39-1. Build the thing that makes the work repeatable. A jig is a purpose-built fixture that holds the work so the cut comes out right every time — a tool whose only job is to make another job correct. A woodworker will spend a morning building a jig to make forty identical cuts in the afternoon, and this is not considered a detour.

Your jigs: the project scaffold, the QC report generator, the figure template, the assertion library, the synthetic-data generator, the one command that reproduces everything. The time to build them is before you need them forty times, and the reason they don't get built is that they look like they aren't the work.

39-2. Sharpening is not time away from the work. Craftspeople maintain tools on a schedule, not when the tool fails, and nobody regards it as procrastination. Environment maintenance, dependency updates, test repair, and documentation are the same category. Schedule them; do not let them become emergencies.

39-3. Everything in place before you begin. Kitchen discipline: assemble and arrange everything before starting, because you cannot go find the thing mid-service. Before an analysis session: data staged, environment active, questions written, the previous state understood. Fifteen minutes at the front saves an hour of context-switching.

39-4. Measure twice, cut once — and know which cuts are irreversible. Most computational operations are reversible and should be done fast and loosely. A few are not: overwriting raw data, sending results to a collaborator, submitting, releasing, making a public claim. ★ Identify the irreversible operations explicitly and apply an entirely different standard of care to them. The failure is applying uniform caution to everything, which is both slow and insufficient where it matters.

39-5. Work with the grain. A woodworker reads the material and cuts with its structure. Data has structure too — hierarchy, sparsity, batch, compositionality, count nature — and methods that fight it produce tear-out. Most methodological disasters are attempts to cut across the grain with more force.

39-6. Quality where nobody looks. The joint that doesn't show is the one that fails. In our work: the code path nobody reviews, the supplementary figure nobody reads, the metadata field nobody checks. These are exactly where errors survive, precisely because they are unobserved.

39-7. Know when it is finished. Craftspeople talk about overworking a piece — continuing past the point of improvement into damage. Analyses can be overworked: the eleventh sensitivity analysis, the fourth reorganization of the figure panel. Have a criterion for done, set in advance, and honor it.

39-8. Sign your work. A maker's mark is an accountability device disguised as pride. In practice: the person who wrote the code is named in it, the person who made the figure is recorded, the person who signed off on the QC is identifiable. Anonymous work is unowned work, and unowned work degrades.
Chapter 40: The Paved Road
Goal. Decide what to build as shared infrastructure, and who owns it. Where the wisdom comes from. Platform engineering, and the paved-road-versus-mandate distinction. What you should walk away with. Four tiers of artifact, the template-versus-library rule, stated support levels, and a quarterly orphan review.

Chapter 39 argued that building a jig is real work. This chapter is about which jigs to build, who owns them, and why most internal libraries in research groups are dead within two years.

40-1. Paved road, guardrail, mandate. Three ways to make a practice happen, and they are not interchangeable:

Paved road — the default path is genuinely easier than the alternative. You may leave it; you are then on your own.
Guardrail — an automatic check catches you when you go wrong. Costs nothing until it fires.
Mandate — you must. Requires authority you do not have (Ch. 38-9).

★ Most internal tooling fails because it is introduced as a mandate while being worse than the thing it replaces. Build the paved road first; the adoption problem then solves itself, because Chapter 38-10 applies — a norm that saves effort immediately requires no authority.

40-2. Four tiers of internal artifact, in ascending order of cost. Most teams reach for tier C when A or B would have done.

Tier
Form
Maintenance cost
Use when
A — Conventions
Written agreements. No code.
Near zero
Naming, directory layout, where things live, ID formats. Highest return per unit effort in the whole handbook.
B — Templates
Copied and then diverged
Zero after creation
Project scaffolds, pipeline skeletons, analysis starters, report formats
C — Libraries
Imported, versioned, tested
Real and permanent
Logic that must be correct and identical everywhere
D — Services
Something running
High; implies on-call
Almost never at this scale. Push to institutional infrastructure.


40-3. Template or library — the distinction people get backwards. ★ If it is structure people will want to modify, make it a template and let it diverge. If it is logic that must be correct and consistent, make it a library and forbid divergence. A pipeline skeleton is a template. The function that computes your normalization is a library. Getting this inverted produces either a library everyone forks in week one, or five silently drifting copies of a calculation that must not drift.

40-4. The build test. Build it only when all four hold:

Done more than three times, by
more than one person, where
variation is a defect rather than a signal of judgment (Ch. 40-8), and
someone will accept ownership.

The fourth is the one that kills things, and it is the one nobody checks.

40-5. A catalogue for a computational team. Roughly in order of return:

Conventions (tier A) — canonical sample and subject identifiers and where they are defined; directory layout; file naming; what goes in raw versus derived versus curated (Ch. 41); branch and release conventions; where decisions are recorded (Ch. 38-1).

Templates (tier B) — project scaffold; workflow skeletons for the standard pipelines, parameterized rather than hard-coded; the analysis starter with preflight already wired in; the accession report (Ch. 7-5); the project charter (Appendix B); the onboarding cold-machine drill (Ch. 37-10).

Libraries (tier C) — I/O and object construction against your canonical schema; the assertion and contract library that enforces Chapter 23-2 at every boundary; QC metric computation and report generation; the figure theme, so that every plot in every paper is consistent without anyone thinking about it; statistical wrappers that encode the group's defaults; synthetic data generators with known ground truth (Ch. 15-2); a benchmarking harness for comparing methods on your own data.

Deliberately not built — anything with one user; anything that wraps a well-maintained external tool without adding judgment; anything whose purpose is to hide a tool people should learn (Ch. 37-4).

40-6. Every artifact carries a support level, stated in its README. Three levels, and publishing them costs nothing:

Supported — named owner, tests, we fix bugs, breaking changes are announced.
Shared — here it is, no promises, read the source before depending on it.
Personal — do not depend on this.

★ The failure mode is not absent ownership; it is implied ownership — people depending on something whose author never agreed to support it, and discovering this during a deadline. Stating the level converts a future argument into a present sentence.

40-7. Ownership is accepted, not assigned, and it is paired. An owner who did not agree is not an owner. Every supported artifact has a named owner and a named second (Ch. 28-1), because the single-owner utility is the purest form of the single-point-of-knowledge defect and it is usually the team lead.

40-8. The quarterly orphan review. List every internal artifact with its usage, its owner, and its last commit. Anything used and unowned is a decision to make, not a fact to note: adopt it, archive it, or replace it. ★ An unowned utility that three projects depend on is a liability with a publication deadline attached to it.

40-9. Sunset deliberately. Archive rather than abandon. A deprecated artifact gets a notice, a migration path, and a date (Ch. 48-4). Silent abandonment is how a group loses trust in its own infrastructure, after which everyone quietly maintains a private copy of everything.

40-10. Budget it, or it will not happen. Chapter 17-5's threshold is the mechanism: when reproducibility failures cross a stated line, new analysis stops until infrastructure is repaired. Without a trigger, infrastructure work is always deferrable and is therefore always deferred — and Chapter 41-2 changed the economics enough that the deferral is now much harder to justify.
Chapter 41: Standardizing the Seams
Goal. Resolve the recurring argument about which tools the team should use. Where the wisdom comes from. Modular platform design and interchange-contract thinking. What you should walk away with. Standardize the seams and not the tools; mandate outputs rather than tools; and a three-tier storage model with identity discipline at its centre.

The recurring argument about which tools the team should use is usually unresolvable as posed, because it conflates two layers that should be governed by opposite rules.

41-1. Two opposite failures, both common.

Everyone uses what they are comfortable with. No shared library is possible, code review across the team stops, handoffs fail, onboarding cost multiplies, and the coverage matrix (Ch. 28-2) becomes a lie because the named secondary cannot actually read the primary's code.
Everyone uses the standard tool even where it is the wrong one. The team acquires a methodological ceiling, reimplements things that already exist elsewhere, and slowly loses the ability to evaluate anything outside the standard. This failure is more comfortable and therefore more durable, because nobody is ever criticized for using the standard tool.

41-2. The resolution is a layer distinction, not a compromise. ★ Standardize the seams, not the tools.

The substrate is singular and non-negotiable. Identifiers, on-disk formats, directory layout, metadata schema, environment specification, orchestration, and the interchange boundary between steps. One way. This is Chapter 38-21's standing rigging.
The analysis layer is plural and chosen per task. Use the tool that is actually best for the specific job — a model that only exists in one language, a specialized aligner, a framework with the right primitives.

The bridge is the interchange contract. If every step reads and writes the same representation with the same identifiers, language pluralism costs almost nothing. If it does not, pluralism costs everything, and the argument people are actually having about R versus Python is really an argument about the absence of a contract.

This is the same principle as Chapter 37-12's common platform with swappable configurations and Chapter 37-18's standard bridging parts: fix the interface, vary the component.

41-3. Mandate the output, never the tool. The requirement is a validated object with the required fields, a reproducible environment, and a passing contract check. How you got there is yours. ★ This resolves most of the argument without anyone having to lose it, and it lets the better tool win on merit over time rather than by decree.

41-4. "Best tool" is four different questions. Ask which one is being argued:

Best for the task — raw capability
Best for the team — who can maintain, review, and inherit it (Ch. 37-11, resupply)
Best for the reader — reviewers, reproducers, the collaborator who wants to run it
Best for the half-life — will this still exist in five years (Ch. 37-8)

A tool that is twenty percent better on capability and maintainable by exactly one person is worse for a team of four. That is not a compromise; it is the correct answer to the actual question.

41-5. The decision rule. ★ Comfort wins on near-ties; capability wins when the gap is categorical. The leadership skill is telling these apart honestly and saying which case you think you are in, with reasons. Some choices genuinely are close, and forcing them costs morale for nothing. Some are not close — modern deep learning is one — and pretending otherwise out of politeness is a disservice to everyone, especially the person being protected from the news.

41-6. Break the standard-tool default with a benchmark, not an argument. Run the comparison on your own data, write it up internally, circulate it. Evidence moves people that advocacy does not, and it also survives your absence. Then attach a date to the standard: "we use X, reviewed in 2026," so that "this is what we use" has an expiry rather than being permanent by inertia.

41-7. Every language boundary needs a named owner. A polyglot pipeline is fine; a polyglot pipeline where nobody owns the handoff is not. Chapter 23-3's explicit transfer applies to code as much as to people.
The substrate: data storage
41-8. Three tiers, governed by different rules.

Tier
Contents
Rule
Raw
As received. Never modified.
Write-once, checksummed, permissions locked, provenance recorded. Nobody edits this, including you.
Derived
Everything regenerable from raw plus code
★ You must be able to delete this without fear. If you cannot, you do not have provenance — you have a cache you are afraid of.
Curated
Annotations, harmonized metadata, manifests, sample mappings
Hand-made and unregenerable. Versioned like source code, not like data.


41-9. Your backup priority is probably inverted. Raw is usually recoverable from the core or the provider. Derived is regenerable by definition. ★ The curated tier is the genuinely irreplaceable one — it embodies months of human judgment and cannot be recomputed — and it is almost always the least protected. Put it in version control, review changes to it, and treat a corrupted annotation as a more serious incident than a lost intermediate.

41-10. Almost every storage disaster is actually an identity disaster. One canonical identifier per entity, defined in exactly one place, with a documented mapping to every external identifier you receive. Everything keys to it. ★ Chapter 23-1's observation that the bug is in the join is the same observation from the other side: the join fails because the identity was never fixed.

41-11. Paths are not metadata. If information exists only in a directory name, it will be lost the first time anything is reorganized. Encode structure in paths for human navigation; encode meaning in a manifest.

41-12. Open formats with independent readers. Columnar formats for tables, chunked array formats for matrices, and the domain's standard object formats. The principle beneath the specifics: ★ a format that only your version of your library can read is not storage, it is a cache. Serialized language-specific objects are the clearest example and the most common.

41-13. Version data by content or by date, never by adjective. Content hashes or dated immutable snapshots with a pointer to current. Anything containing final, v2, real, or use_this is an unrecoverable position waiting to happen.

41-14. Tier by cost and deaccession on schedule. Hot, warm, cold, archive, with a stated retirement policy (Ch. 7-9). Storage grows monotonically unless something is designed to shrink it, and the growth is invisible until it is a budget line.

41-15. Write the storage conventions down as tier-A conventions (Ch. 40-2) before you write any code. This is the highest-return document your team will produce, it costs an afternoon, and almost no group has one.

[EXAMPLE — TODO] Placeholder, replace with your own: "Two cohorts merged on a sample ID that meant subject in one and aliquot in the other; the join silently duplicated rows and the effect size doubled."

Public cases that fit. The Mars Climate Orbiter was lost in 1999 because one team supplied impulse values in pound-force seconds while the receiving software expected newton-seconds — an interface contract failure, not a coding error. Ariane 5 Flight 501 was destroyed in 1996 because inertial reference software carried over from Ariane 4 was run outside the flight envelope it had been validated for. Both are seam failures at a component boundary that each side believed it understood.
Chapter 42: The Workmanship of Code
Goal. Explain why craft standards are an operational requirement at this team size. Where the wisdom comes from. David Pye's theory of workmanship, and the tradition of finishing the parts nobody sees. What you should walk away with. Risk versus certainty as a description of your own work, the reason the unseen part is the load-bearing part, and ugliness as a technical finding.

This chapter is theory, and it earns its place because a three-person team has no other quality mechanism. Large software organizations substitute process for craft: review boards, QA, staged rollout, dedicated testers. A small scientific team has none of that, which means the standard of the individual maker is the only thing standing between the group and a wrong result. That makes craft psychology an operational concern rather than an aesthetic one.

42-1. Pye's distinction. In The Nature and Art of Workmanship (1968), David Pye separated the workmanship of risk — work in which the quality of the result is not predetermined but depends on the judgment, dexterity, and care the maker exercises while working — from the workmanship of certainty, found in mass production and automation, where the quality of the result is fixed before anything is made.

Pye's crucial refinement: this is not a distinction between hand and machine. Risk-workmanship can use any tool, including power tools. What defines it is that the outcome is still in play while the work is happening.

42-2. Your work is both, and the craft act is moving work from one to the other. Exploratory analysis is workmanship of risk in its purest form: the quality of the result depends entirely on the judgment of the person doing it, moment to moment, and a lapse of attention produces a wrong answer that looks like a right one. A released pipeline is workmanship of certainty: the judgment has been concentrated and stored up beforehand, and the output is predetermined.

★ Building a jig (Ch. 45-1) is precisely the act of converting risk-workmanship into certainty-workmanship. So is writing a test, fixing a seed, pinning an environment, and building a template. Pye's own example is printing: the risk is not eliminated, it is moved earlier and concentrated into the making of the type. This is the cleanest available theory of why infrastructure work is real work rather than a detour from it.

The judgment that matters is knowing which mode a given piece of work should be in. Forcing exploration into certainty produces slow, over-engineered, dead analysis. Leaving a published method in risk-mode produces results only its author can reproduce.

42-3. The unseen parts. Pye's moral claim is that a good workman feels responsible for the durability of what they make, and therefore feels bound to make the unseen parts of the job at least as sound as the visible ones. The same instinct is the one Jobs is remembered for — insisting the interior of a machine be beautiful, on the argument from his father's cabinetry that you use good wood on the back of the fence even though nobody will see it.

For code there is a practical argument that is stronger than the moral one. ★ In software and analysis, the unseen part is the load-bearing part. The hidden joint is what fails. The code path nobody reviews, the supplementary figure nobody reads, the metadata field nobody checks, the preprocessing step everyone inherited and nobody re-derived — these are exactly where errors survive, because they are unobserved. Craftsmanly attention to invisible work is not sentiment; it is where your defect density actually lives.

42-4. Design and workmanship are different things, and only one of them can be specified. Pye: design is what can be conveyed in words and drawings; workmanship is what cannot. You can specify the API, the schema, the module boundaries, the interface contract. You cannot specify the taste — the choice of decomposition, the naming, the judgment about what to make explicit, the sense of when an abstraction is carrying its weight. That part transmits only by working alongside someone who has it (Ch. 29-5), which is why pairing is a quality mechanism and not merely a redundancy mechanism.

42-5. Ugliness is a technical finding. When a piece of code feels wrong to write — when the function needs seven arguments, when the same transformation appears in four places with small differences, when you cannot name the thing you just built — that discomfort is usually information about the abstraction, not about your mood. ★ "This is hideous" is a legitimate and often correct technical report. Train people to say it and to investigate it rather than to push through. Aesthetic discomfort in a codebase is a leading indicator; test failures are a lagging one.

42-6. Legibility is the actual virtue, and it has an audience of two. Not elegance for its own sake. Analysis code is read by exactly two people: the person who wrote it, six months later, with no memory of it, and one colleague trying to reproduce a number under deadline. Both are hostile readers with no context. Everything that reads as craft — consistent naming, one obvious path, explicit rather than clever, small honest functions, the transformation visible rather than buried — is in service of those two readers. Write for them and the beauty follows; write for beauty and you will produce something admirable and unusable.

42-7. Honest construction. Architecture has a long argument about expressing structure rather than concealing it. The code version: do not hide where the judgment happened. The threshold that was chosen, the samples that were dropped, the parameter that was tuned — these should be visible in the code at the point of use, not buried three modules down in a defaults file. A pipeline that conceals its own decision points is dishonest construction even when every decision was correct.

42-8. Where variation is craft and where it is defect. Pye argued that good workmanship imparts valuable diversity. Be careful importing that. In a codebase, variation in convention — structure, naming, formatting, project layout — is pure cost and should be standardized ruthlessly (Ch. 44-3). Variation in approach — how a problem is decomposed, what abstraction is chosen, what the analysis is shaped like — is where individual judgment is legitimate and valuable. Standardize the container; leave the contents to the maker.

42-9. The failure modes of the craft disposition. Pye was explicit that he was not arguing risk-workmanship is always or necessarily valuable, and the caution applies:

Gold-plating. Refining an artifact past the point where refinement changes any outcome. Chapter 39-7.
The beautiful abstraction that serves nobody. A framework built for generality that will only ever have one user. This is a craft impulse pointed at the wrong object.
Refactoring as avoidance. Cleaning code is legible, satisfying, and produces visible progress. It is also the most comfortable way to not do the hard thinking, and it is nearly undetectable from the outside.
Craft applied to throwaway work. An exploratory notebook does not need to be beautiful. Knowing which register you are in (Ch. 10) is itself part of the craft.

42-10. The standard, stated plainly. For anything that will be read again or run by someone else: it should be possible for a competent stranger to reproduce your result from your repository, in an afternoon, without asking you a question. That is the whole of it. Every practice in Part VIII exists to make it achievable, and it is a craft standard rather than a compliance standard because in a team this size nobody is going to check.
Chapter 43: Machine Assistance
Goal. Answer the question you are actually being asked about machine assistance. Where the wisdom comes from. Cost-structure reasoning, plus Pye's vocabulary applied to a new hybrid. What you should walk away with. The reframe from usage to reallocation, the generation-verification asymmetry, monitor rather than pilot, and indicators worth reporting.

Written in 2026, with the explicit caveat that this chapter has the shortest half-life in the handbook. The specific capabilities will change; the structural argument should survive, because it is about cost asymmetries rather than about any particular system.

43-1. The question you will be asked is malformed, and the reframe is the answer. "Is the team using AI effectively?" has no answer as posed, because effectively is undefined and the obvious metrics — hours saved, proportion of code generated, tool adoption — measure consumption rather than output. ★ The right question is whether the freed capacity was reallocated to work that was previously unaffordable. A team that generates the same output faster has captured none of the value. A team that now builds the tooling, runs the sensitivity analyses, and tests the hypotheses it used to skip has captured all of it.

43-2. What actually changed, mechanically. Four things, and only four:

The first draft is nearly free. Of code, prose, a plan, a literature scan, a schema. This matters most where the first draft was the barrier to starting.
Bad versions are cheap. Chapter 48-9 said to make the first implementation embarrassing and fast; that is now trivial rather than merely advisable.
The iteration loop is shorter (Ch. 17-1), which is the master variable, and therefore this is the largest legitimate gain.
★ Jigs became affordable. Chapter 39-1 argued that a morning spent building a fixture to make forty correct cuts is not a detour. When the fixture costs twenty minutes instead of a morning, the entire calculus of infrastructure changes, and a great deal of tooling that was correctly deferred for a decade is now correctly built. If a team has adopted these systems and its jig count has not gone up, it is using them for the wrong thing.

43-3. What did not change: verification. ★ Generation got cheap; checking did not. The binding constraint therefore moves from production to review, and a team that scales the first without scaling the second is manufacturing unverified artifacts at speed. This single asymmetry generates nearly every practice below.

43-4. The Pye problem. Chapter 42-1 separated the workmanship of risk — where quality depends on the maker's judgment during the work — from the workmanship of certainty, where quality is fixed in advance. Machine assistance produces a dangerous hybrid: it has the feel of certainty and the reality of risk. The output arrives fast, fluent, confident, and formatted, which are the surface signals of predetermined quality, while the actual correctness remains entirely contingent. Every instinct that tells a craftsperson "this was made carefully" is being triggered by something that carries no such guarantee.

The operational consequence: fluency is not evidence. Train the team to notice that the confidence of the output is uninformative, because it is the one heuristic everyone has and it no longer works.

43-5. Map it onto the gears (Ch. 10).

Gear
Policy
Fast — exploration, throwaway, consequences near zero
Use freely. This is where the gains are largest and the risk is genuinely near zero.
Checked — wrong answers cost time
Use with tests that were written independently of the generated code. A generated implementation checked by a generated test is not checked.
Verified — output reaches other people
★ Generated code entering a verified artifact requires more review than human code, not less, because there is no author who understands it. Nobody has the model in their head.


43-6. The line that matters most: implementation versus decision. Use it to implement; never to decide. Which samples to exclude, which model is appropriate, what the effect means, whether the result is real, what question to ask — these are the parts that constitute the science and the parts where a fluent wrong answer is undetectable. ★ Nothing in the methods, results, or interpretation of a paper should exist that no human on the author list derived.

43-7. The strongest use is as monitor, not as pilot. Chapter 29-2 argued that the monitoring role is the safety-critical one and the undervalued one. Machine assistance is unusually good at it and unusually safe there, because verification of a critique is cheap while verification of a construction is expensive. Find the bug. What is wrong with this argument. What did I not consider. What would a hostile reviewer attack. Give me three alternative explanations for this result. That last one is Chapter 47-4's differential diagnosis, and it is now available on demand — probably the highest-value scientific use currently available.

43-8. It is a confirmation machine if you let it be. Ask a leading question and you will get agreement, fluently and at length. This interacts badly with Chapter 9-2: the collaborator who wants X equals Y now has a tool that will help them believe it. ★ Ask for the case against, never for the case for. Make this a stated team norm, because the failure is invisible from the inside.

43-9. Provenance and declaration. Adopt the Chapter 6-2 discipline: declare the mode. For anything that leaves the team, the artifact should carry whether it was generated and whether it was reviewed line by line. Log prompts for anything consequential. Chapter 39-8 said anonymous work is unowned work and unowned work degrades — that hazard is now much easier to create by accident.

43-10. Blinding survives, and matters more. Chapter 18-2's discipline is unchanged: if a system has seen the outcome, do not use it to tune the analysis. The temptation is larger now because the tuning is so cheap.

43-11. Do not let it write the paper. Not for integrity reasons alone, though those hold. Chapter 2-12: writing is thinking, and the gaps in the prose are the gaps in the work. ★ Generating the paragraph you could not write means you never find out that you did not have the result. Use it to edit, to tighten, to check for unclear antecedents, to challenge the argument. Not to produce the argument.

43-12. Indicators that actually answer the question. When asked whether the team is using these systems effectively, report against these rather than against usage:

Loop count. Iterations per week on live questions (Ch. 17-1). Up, or nothing happened.
Jig count. Pieces of durable internal tooling built this quarter that previously would not have been worth building.
Verification capacity. Test coverage, review throughput, and reproducibility failures — has checking scaled with generating, or has the ratio degraded?
Exploration breadth. Hypotheses and sensitivity analyses run that would previously have been skipped as too expensive.
Rejected-approach documentation (Ch. 14-6), which is now cheap and was always valuable.
Junior competence trajectory (Ch. 47), which is the one that will be missed. See the next chapter.

★ The honest summary sentence, if you need one: "we generate more and we verify more; here is the ratio." A team whose generation went up and whose verification did not has a finding to report, not an achievement.
PART IX — THINKING
Chapter 44: Fatigue and Attention
Goal. Treat the analyst as an instrument with operating conditions. Where the wisdom comes from. Human factors research on stress and attention. What you should walk away with. Rest as force preservation, and the recognition that attention narrows physiologically rather than as a matter of will.

44-1. You do not rise to the occasion. Under deadline pressure nobody carefully reasons to the correct statistical test; they execute what is habitual. This is the whole argument for making the correct thing the habitual thing.

44-2. Attention narrows physiologically under stress. Tunnel vision, loss of peripheral awareness, degraded working memory, time distortion. You stop seeing the rest of the system. This is not a character weakness and cannot be willed away; checklists exist precisely because attention narrows.

44-6. Sleep, food, and recovery are operational. High-consequence professions treat rest as capability maintenance, not as personal indulgence, and they are right. A tired analyst is an unreliable instrument. ★ Research culture tends to invert this and calls the inversion dedication.
Chapter 45: Baseline and Anomaly
Goal. Detect anomalies by first establishing what normal looks like. Where the wisdom comes from. Behavioural baselining, adapted to data. What you should walk away with. Anomalies run in both directions, three converging indicators are signal, and the threshold must be set before you look.

45-1. You cannot detect an anomaly without a baseline. Establish deliberately what normal looks like for this dataset, this run, this instrument, this batch — before hunting for signal.

45-2. Anomalies run in both directions.

Above baseline — something present that should not be. A cluster too clean. An effect larger than anything published. A p-value smaller than the sample size supports. Perfect separation. ★ Extraordinary signal is usually a technical artifact in costume.
Below baseline — something absent that should be present. No batch effect across three runs. A canonical marker missing from a cell type you are confident about. Suspiciously complete metadata. ★ This is the more valuable half and the more commonly missed. Absence usually means someone upstream acted on information you do not have.

45-3. Where to look.

Layer
What to baseline
Distributions
QC metric shapes, not just their means
Technical signatures
Contamination, doublets, depth, dropout, batch markers
Relationships
Associations between things that should be independent — batch with condition, technician with outcome
Structure
How observations distribute across the embedding or feature space
Provenance
Metadata fields, naming conventions, file dates, who touched what
Overall feel
The sense that something is off, before you can articulate it — this is real information and should be spoken aloud, not suppressed until it is defensible


45-4. One anomaly is noise; three converging is signal. Applies to the same sample, batch, or module.

45-5. Decide in advance what you will do about it. ★ The failure mode is not missing the anomaly. It is noticing it, rationalizing it, and proceeding. Set the threshold before you look, so the anomaly triggers an action rather than a negotiation with yourself.
Chapter 46: Pattern Recognition and Its Failures
Goal. Understand how expert pattern-matching works and how it fails. Where the wisdom comes from. Clinical diagnostic reasoning, which has been measuring its own errors for forty years. What you should walk away with. The named error modes, of which diagnostic momentum is the one that should alarm computational biologists, plus their countermeasures.

Clinical medicine has thought harder about this than any other profession, because it has been measuring its diagnostic errors for forty years. The accumulated finding is that diagnostic errors are mainly cognitive rather than knowledge-based, and that risk rises under time pressure when practitioners use shortcuts. Notably, as clinicians gain experience, error more often arises from flawed reasoning process than from gaps in knowledge. That last point should worry every senior scientist.

46-1. Two systems, and knowing which you are in. Fast pattern-matched recognition and slow analytical reasoning. Expertise is largely a library of compiled patterns — the senior person who glances at an embedding and says "that's ambient contamination" is matching, not deriving. This is efficient and it is correct most of the time.

46-2. The failure mode is specific: confident wrong matches on novel situations. New modality, new assay, unfamiliar disease context, unfamiliar instrument — exactly where your pattern library will mislead you with full confidence. The skill is recognizing that you are outside your compiled experience and deliberately switching to slow mode.

46-3. The named errors, with their computational forms.

Premature closure — treating a working hypothesis as final and not pursuing further workup. The most common diagnostic error in medicine and in data analysis. You have an explanation, it fits, you stop looking.
Anchoring — clinging to an initial impression even as conflicting data accumulate. The first hypothesis you formed about the dataset is still steering you in month four.
Diagnostic momentum — ★ the one that should alarm computational biologists most. A label initially recorded as merely probable gets propagated through subsequent notes and becomes increasingly entrenched as correct, with later specialists accepting the prior working diagnosis without independently reviewing the data. This is exactly what happens to a provisional cell type annotation. It is assigned as a guess in week two, inherited by eight downstream analyses, cited in the figure legend, and by month six it is ground truth that nobody has re-derived. Diagnostic momentum in annotation is probably the largest unexamined source of error in single-cell and spatial biology.
Search satisficing — finding something and stopping. You found a batch effect; you did not check whether there were three.
Availability — over-weighting what you have seen recently. The method from last month's paper is now your default explanation.
Confirmation bias — collecting evidence for rather than testing against.

46-4. The countermeasures, all cheap.

Generate the differential first. Before committing to an explanation, write down three alternatives, explicitly, in the notebook. This is the single most effective debiasing practice in clinical education and it transfers directly.
Take a deliberate timeout. A scheduled pause partway through an analysis with one question: what else could produce this?
Ask what the most dangerous explanation would be. Not the most likely — the one that would be worst if true, and which you would most regret missing. Usually a technical confound.
Re-derive inherited labels. ★ Once per project, independently regenerate one annotation or classification you inherited. Find out whether it holds.
Get a second read from someone with no stake in the outcome.
Distinguish the working hypothesis from the confirmed one, in writing, and mark every place the provisional version is being used as if it were final.

46-5. Cases are how this is taught. Medicine runs morbidity and mortality conferences; engineering studies structural failures as a formal discipline. Both professions maintain institutional memory of their own disasters, and research groups almost never do. Chapter 49-4.
Designing against your own bias
46-6. Do not rely on resisting bias; make it structurally difficult. Chapter 18-2 covers blinding, which is the strongest available tool and dramatically underused outside physics.

46-7. Pre-specify. Write the analysis plan and the figure legends before unblinding, and commit to reporting whatever emerges. This is not bureaucracy; it is the only thing that distinguishes a confirmatory result from an exploratory one, and the distinction is real regardless of whether anyone makes you declare it.

46-8. Separate exploration from confirmation, explicitly and in the paper. Both are legitimate. Presenting the first as the second is the most common form of quiet misconduct in computational science, and it is usually committed by people who would be horrified to hear it described that way.

46-9. Adversarial review inside the team. Assign someone the explicit job of trying to break the result before submission, and give them time to do it. Rotate the role. Reward finding things.

46-10. Ask what would change your mind, out loud, before you look. If nothing would, you are not doing science on this question, and it is better to notice that early.
Chapter 47: Judgment and Automation
Goal. Consider what machine assistance does to people over years. Where the wisdom comes from. Automation complacency research from aviation, and skill-acquisition theory. What you should walk away with. Struggle on the training set, assist on the production set, and the argument that tacit knowledge is now the moat.

Chapter 43 covered practice. This chapter covers what happens to the people, over years, and it is the part that will not show up in any quarterly metric until it is too late to fix cheaply.

47-1. Expertise is compiled repetitions, and the compiler runs on struggle. Chapter 46 described the pattern library — the senior person who glances at an embedding and says that is ambient contamination is matching against thousands of stored instances. Chapter 30-6 noted that repetitions build expertise only when coupled to feedback. ★ The first draft is where the compilation happens. A junior who never writes the wrong version, sees it fail, and works out why does not acquire the pattern; they acquire the ability to obtain a plausible answer, which is a different and much shallower skill that looks identical from the outside for about three years.

47-2. This is the most consequential open risk in the handbook and there is no evidence base yet. It should be treated as a live hazard managed by policy rather than as a settled problem. State the uncertainty honestly to your team; do not pretend either that it is fine or that it is catastrophic.

47-3. The practical rule: struggle on the training set, assist on the production set. For work whose purpose is the artifact, use everything available. For work whose purpose is the person — a first implementation of a method they will own, a debugging session, a derivation, a first analysis of a new modality — the struggle is the deliverable, and shortcutting it destroys the thing being purchased. Name which mode a task is in when you assign it. ★ "This one you write yourself, and I want to see the version that does not work" is a legitimate and increasingly necessary instruction.

47-4. Practice pieces become more important, not less. Chapter 30-4 recommended making things you intend to discard — reimplementing a published method from scratch, analyzing a dataset with a known answer. These now have no external justification whatsoever, which is precisely why they must be scheduled rather than hoped for.

47-5. Tacit knowledge is the moat and it is unautomatable by construction. Pye's distinction (Ch. 45-4): design is what can be conveyed in words and drawings; workmanship is what cannot. What cannot be specified cannot be prompted. The judgment about which decomposition to choose, when an abstraction is carrying its weight, which anomaly is worth chasing, when a collaborator is describing their design inaccurately — none of this transmits through text, which means it transmits only through Chapter 27-5 and Chapter 30-3. Pairing and apprenticeship move from good practice to the primary mechanism by which the team continues to exist.

47-6. Diagnostic momentum accelerates. Chapter 47-3 — a provisional label propagating until it is treated as established — gets faster when the provisional label can be elaborated, defended, and built upon in seconds. ★ The countermeasure is unchanged and now mandatory: once per project, independently re-derive one inherited assumption, by hand, without assistance.

47-7. Watch for the loss of the naive question. Chapter 2-14 called "sorry, why do we do it that way?" the most valuable sentence in a group meeting. It is now much easier to obtain a confident-sounding answer privately, in ten seconds, without exposing that you did not know. That is a real convenience and a real loss, because the question asked out loud was never only for the asker — it was the mechanism by which the group discovered that nobody actually knew.

47-8. Automation complacency is a studied phenomenon, not a novel worry. Aviation has decades of evidence: as automation reliability increases, monitoring performance degrades, and the operator's ability to detect the automation's rare failures declines precisely because those failures are rare. The failure mode is not the automation being wrong; it is the human having stopped looking. Chapter 29-2's remedy applies directly — the monitoring role must be assigned, practiced, and verbalized rather than assumed.

47-9. Chapter 2 is now a competitive statement rather than a sentimental one. Anomaly appetite. Cultivated confusion — sitting in not-knowing without collapsing to an answer, which is exactly the pressure these systems relieve and exactly the discomfort that produces original work. Taste. Problem selection (Ch. 10). Knowing which of ten possible papers matters (Ch. 9-6). ★ The scarce faculties are now the ones the machine cannot supply, which means the ethos chapter is no longer the soft one at the front — it is the description of where the remaining advantage lives.

47-10. The test to apply to yourself, annually. Take a problem in your own domain that you would ordinarily delegate to a machine, and solve it without one, and notice how it feels. If it feels like a language you used to speak, that is the finding, and it is early enough to act on.
Chapter 48: Creativity as Practice
Goal. Give creativity the machinery it needs rather than treating it as a mood. Where the wisdom comes from. Practice traditions across the arts and sciences. What you should walk away with. Separated generation and evaluation, question and anomaly logs, out-of-field reading on a schedule.

Chapter 2 is the disposition. This is the machinery.

48-1. Separate generation from evaluation, in time. The most common creativity failure in expert teams is evaluating too early — a half-formed idea is met with a correct objection and dies. Run explicit generation sessions where objections are deferred, and separate evaluation sessions afterward. ★ Expert groups are especially bad at this, because their objections are usually right.

48-2. Keep a question log, per person, visible. Questions you cannot currently answer. Reviewed monthly. Most will be nothing; a few will become projects when a new method or dataset makes them attackable, and you will not remember them otherwise.

48-3. Keep an anomaly log. Things that did not fit and were set aside. Same review cadence. ★ Anomalies that were individually dismissible frequently become a pattern in aggregate, and the aggregate is only visible if you wrote them down.

48-4. Read out of field on a schedule. One paper a week from somewhere you have no business being. Most methodological novelty is transfer, and transfer requires having the source structure in your head. Discuss one of these in group meeting monthly, with the explicit question: does anything here apply to us?

48-7. Steal the structure, not the content. When you read something good outside your field, extract the shape of the argument or the method — what it did structurally — rather than its subject matter. That shape is the transferable part.

48-8. Protect the unassigned fraction. Chapter 2-6. A named, defended, unjustified allocation of time. It will be the first thing cut and it should be the last.

48-9. Make bad versions quickly. The first implementation should be embarrassing and fast. Elaborate planning before a first attempt is usually avoidance, and the first attempt teaches you what the plan should have said.

48-10. Talk to people who cannot follow you. Explaining a problem to someone outside your specialty forces the reformulation that produces insight. This is why the naive question is valuable (2-14) and why the person who asks it should be thanked.

48-11. Sit with the problem before reaching for a method. The instinct to immediately apply a familiar approach short-circuits the phase where you understand what the problem actually is. Give it a day. Look at the raw data by hand. Plot it badly.

48-12. Creativity has a maintenance requirement. It degrades under chronic overload, fragmented attention, and fear. A group operating permanently at capacity will be reliably productive and reliably unoriginal, and will not be able to tell.


PART X — LEARNING
Chapter 49: Reviews
Goal. Convert experience into learning, which does not happen automatically. Where the wisdom comes from. After-action review practice, and the institutional memory of failure kept by engineering and medicine. What you should walk away with. Four questions, blamelessness enforced by behaviour, reviews of successes, and an output that changes an artifact.

49-1. Four questions. What was supposed to happen. What actually happened. Why the difference. What changes.

49-2. Blameless, and enforced by leadership behavior. Chapter 23-5 and 23-6.

49-3. Review successes, not just failures. ★ Wins hide luck. Run one on the paper that got accepted, especially that one. Unexamined success teaches the wrong lesson and the team repeats it until it fails.

49-4. Keep an institutional memory of failure. Engineering formally studies its collapses; medicine holds morbidity and mortality conferences. Both maintain a shared, non-punitive record of what went wrong and why, and both consider it foundational to professional training. Research groups almost universally do not, which is why every generation of students rediscovers the same errors. Keep a file. Read it with new arrivals.

49-5. Name what to sustain as explicitly as what to change. Teams that only enumerate faults stop volunteering material.

49-6. The output is a changed artifact. If the review did not change a template, a checklist, a default, or a standard response, it did not happen.

49-7. Experience does not automatically become learning. The review is the mechanism that converts repetitions into the pattern library of Chapter 46. Without it you accumulate years of unexamined habit and call it seniority.
Chapter 50: Standards and Improvement
Goal. Make improvement possible by making practice consistent. Where the wisdom comes from. Lean manufacturing, honestly attributed. What you should walk away with. Standardization as the precondition for improvement, root-cause discipline, and deliberate deprecation.

50-1. Standardization is what makes improvement possible. You cannot improve a process that varies randomly between people and weeks. This — not tidiness — is the argument for templates and conventions.

50-2. Find root causes, not symptoms. Ask why repeatedly. And go look at the actual thing: the raw file, the log, the instrument output — not somebody's summary of it. Reading a summary is how you inherit the summarizer's assumptions.

50-3. Decide deliberately; implement fast. Consider the alternatives properly, build agreement, then move quickly. The research failure mode is the inverse: a fast decision, a slow implementation, and endless relitigation.

50-4. Deprecate on purpose. Software engineering has conventions for retiring things — deprecation notices, migration paths, version guarantees. Scientific software mostly does not, and the result is that nobody trusts it. If people depend on your package, say what you promise and how you will warn them.

50-5. Small reversible changes over large irreversible ones. Frequent small merges beat long-lived branches. Frequent small releases beat annual ones. This is settled in software engineering and largely unadopted in research code.

50-6. Anything shared has a maintenance cost, forever. Decide, before releasing, whether you are paying it. "Released and unmaintained" is a legitimate choice if it is stated; it is corrosive if it is discovered.
Chapter 51: Your Own Practice
Goal. Apply the handbook to the person running the team, since nobody else will. Where the wisdom comes from. Everything above, at n=1. What you should walk away with. The router trap, the perishable skill worth protecting, invisible work made visible, and one line of work that is yours.

Every other chapter is about the team. This one is about the person leading it, because in this role nobody else will raise any of it, and because the failure modes are specific and largely invisible from inside.

51-1. Your calendar is the team's binding constraint, and you are the only person who can see it. In a group of four where one person holds the external relationships, the technical standards, and the escalation path, that person's availability caps everything. ★ Treat your own fragmentation as a team-level defect rather than a personal inconvenience, because that is what it is, and it is the framing that lets you act on it without it sounding like a complaint.

51-2. The specific trap of this role is becoming a router. Every message is answerable by you and only by you, each answer takes four minutes, and after two years you have not done a piece of sustained technical work in months. It happens gradually and it feels like being useful. The diagnostic: when did you last spend three uninterrupted hours on one problem? If the answer is measured in months, you have become infrastructure, and infrastructure does not get promoted, cited, or renewed.

51-3. Stay technical enough to judge. Chapter 5-7 said to lead from one level above the work — but in this role, unlike a PI's, remaining operational is the point (Preface §0.4). The standard is not "still the best programmer." It is: can you still tell, unaided, whether a result is right? That is the perishable skill and it perishes quietly. Protect enough hands-on work to keep it, and be explicit with yourself that this is a professional requirement rather than a preference.

51-4. Apply your own chapters to yourself. Most of the handbook works at n=1:

Run a project charter (Appendix B) on your own year. What is the question, what would change the plan, what are the stopping criteria.
Set your own capacity threshold (Ch. 8-5) and log when you cross it.
Run an after-action review on your own quarter, blamelessly, in writing (Ch. 49).
Maintain your coverage matrix (Ch. 28-2) with yourself in it, honestly. Which functions have you as the only owner?
Keep the question log and the anomaly log (Ch. 48-2, 53-3). These are the first things to lapse and the first things you will miss.

51-5. Your development has no owner. A postdoc has a mentor; a PI has a promotion track and a committee. ★ Nobody is planning your growth, and the institution's default is to keep you doing what you are already good at. Decide annually what capability you are adding — a method, a domain, a technology, a form of writing — and schedule it the way you would schedule a collaborator's deliverable. It will not otherwise happen.

51-6. Make your invisible work visible, once a quarter, in writing. The accession assessments (Ch. 7-7), the declines, the escalations absorbed, the infrastructure built, the people developed, the collaborations rescued. Not for vanity — because none of it appears in a publication record, and the people evaluating you are working from a publication record. A short quarterly note to your PI listing what was carried is the single highest-return administrative habit in this role.

51-7. Keep one thing that is yours. One line of work you own scientifically, that would continue if the collaborations stopped, and that has your name first on it. This is what distinguishes a scientist from a service. It will always be the thing that slips, because it has no external deadline and no one asking, which is exactly why it needs a protected allocation on the calendar (Ch. 2-6).

51-8. Reef early applies to you (Ch. 36-19). The moment you first wonder whether you are carrying too much is the moment to reduce, not the moment you are certain. By the time it is obvious, the reduction is more expensive and more visible.

51-9. Read the indicators in Chapter 26 with yourself as the subject. Effort shifted from advancing to defending. The same decisions relitigated. Working at night without producing anything. Sunk-cost language. Those indicators diagnose dead projects; they diagnose dead roles the same way, and the honest thing is to check occasionally.

51-10. You are training your replacement whether or not you intend to. Everything about how you work — what you tolerate, what you check, whether you say "I do not know," how you respond to being told you are wrong — is being learned by three people who have no other model for this job, because there is not one. ★ You are, in a small way, writing the profession's craft knowledge by demonstrating it, since the first cohort in this role is still in it and nothing has been written down. That is a genuine responsibility and also the most interesting thing about the position.
Chapter 52: What Not to Borrow
Goal. Name what will feel most compelling exactly when it is most harmful. Where the wisdom comes from. The pathologies of every profession borrowed from here. What you should walk away with. Nine things to leave behind, of which manufactured urgency is the most damaging and the aesthetic itself is the most seductive.

Every profession this handbook draws from has pathologies alongside its practices. These are the ones that will feel most compelling exactly when they are most harmful.

52-1. Manufactured urgency. From operational professions, where it is real. Almost no analytical decision is time-critical and most are reversible. Importing urgency produces reckless commitment and burnout, and it is the most damaging borrowing available.

52-2. Suffering as evidence of seriousness. From every high-status profession that has ever existed. Research runs on years, not days. Endurance means pace, not intensity, and Chapter 25 depends entirely on people being able to say "this isn't working" out loud.

52-3. Process as a substitute for judgment. From large engineering organizations. Every checklist in this handbook exists to free attention for the parts that require thinking. A team that follows all of it and thinks about none of it has extracted the wrong thing.

52-4. Optimization pressure. From software engineering and manufacturing, where the objective is known. Optimizing throughput on a fixed objective is the correct move in production and the wrong one in research, where the value is concentrated in objectives you have not identified yet.

52-5. Diagnostic confidence. From clinical medicine, which needs it because a decision must be made today with the patient in front of you. You usually have more time. Take it.

52-6. Perfectionism. From the craft trades, where the object is finished and permanent. Analyses are not finished; they are abandoned at an appropriate point. Chapter 39-7.

52-7. Hierarchy. Every operational profession assumes lawful authority. You have none. Everything in Part VI works only through credibility and voluntary cooperation, which is why Chapter 30 is longer than Chapter 4.

52-8. Any framing that makes the work feel more dramatic rather than more organized. ★ The value here is checklists, stopping rules, named owners, pre-set thresholds, and structured reports. If you have those and have discarded every borrowed word, you have taken everything worth taking.


APPENDICES
Appendix A — Quick Reference
Before starting

Has the data been accessioned, and at what tier? (Ch. 7)
What kind of work is this? (Ch. 5)
What mode? (Ch. 6)
What engagement tier, and what does it displace? (Ch. 8)
What shape is the ask, and what shape is the data? (Ch. 9) — if "prove X equals Y," ask what it would mean if the answer is no, before agreeing
What gear — fast, checked, or verified? (Ch. 10)
Why are we the ones who win this? (Ch. 11)
Question, in one falsifiable sentence
The three to five results that would change the plan
Stopping criteria, with a date
Fallbacks for compute, data, comms, method, people
Known-good state, named
Figures sketched, legends written
Pilot flying and pilot monitoring assigned (Ch. 30-1)
Authorship criteria agreed, in writing

On an unexpected result Reproduce → freeze → report → reduce → owner decides

Before anything irreversible Stabilized-approach gates met (Ch. 30-5) → preflight checklist, challenge-response, two people → go/no-go poll around the room, out loud, by name. If the gates are not met at the must gate, it slips.

In a crisis Contain → preserve → unblock → recovery point → correctness → warn downstream

Escalating Reach (speed, anyone available) or depth (capability, proper channel)? Then send the alert before the write-up.

After anything Status report → four review questions → change one artifact
Appendix B — Project Charter
One page. Signed by team lead and PI. Reviewed on schedule.

PROJECT:

KIND OF WORK (Ch. 5):              MODE (Ch. 6):

ENGAGEMENT TIER (Ch. 8):           DISPLACES:

ASK SHAPE (Ch. 9):  well-formed / confirmation-seeking / unspecified

DATA SHAPE (Ch. 9): rich (papers ranked below) / adequate / null-so-far

  If confirmation-seeking — what would it mean if the answer is no:

  If rich — question order, and what we give away:

  If null — positive control run:            Detectable effect size D:

ACCESSION TIER (Ch. 7):            GEAR (Ch. 10):

QUESTION (one falsifiable sentence, with purpose):

RESULTS THAT WOULD CHANGE THIS PLAN:

  1.

  2.

  3.

PROCEED CRITERIA (to pass phase 1):

HALT CRITERIA (pause pending decision):

STOP CRITERIA — we stop if, by [DATE], we have not [OBSERVABLE]:

RESUMPTION CONDITION (if held):

PEOPLE

  Decision owner:

  Work owner / partner:

  Standards owner:

COLLABORATION (per partner)

  Their goal:                       Their constraint:

  Their decision maker:             Their timeline:

  What "good" means to them:

  Support intent (Ch. 6-7):

  Exit terms — deliver / leave by / hand over / will not maintain:

  Data agreement:                   Authorship criteria:

VERIFICATION

  Blinding strategy (Ch. 23-2):

  Known-good state:

  Margin — what would have to change for this to flip:

IF THIS FAILS, WE EXTRACT:

  Method:            Negative result:

  Tooling:           People:

REVIEW DATES:
Appendix C — Intake One-Pager
Chapter 8-7. Every request, regardless of size, before it enters the queue.

REQUESTER:                         DATE:

QUESTION (one sentence):

DATA

  Exists / expected by:

  Accession status (Ch. 7):        Assessment report:

DEADLINE:                          DRIVEN BY:

TIER REQUESTED (Ch. 8-2):  0 consultation  1 exploration  2 analysis  3 co-development

WHO DOES THE WORK ON THEIR SIDE:

IF THE ANSWER IS NO, WHAT HAPPENS:

--- team use ---

CURRENT TIER 3 COUNT:              CAPACITY THRESHOLD (Ch. 8-5):

DISPLACES:                         OR: QUEUED, EXPECTED START:

DECISION:                          BY:
Appendix D — Coverage Matrix
Chapter 27-2. Complete quarterly. Every row with a blank secondary is a declared risk, escalated in writing.
Appendix E — A Statement of Values
Aerospace mission operations maintains a short statement of professional values, unchanged for sixty years, posted on the wall. It has held up better than most such documents because each item is defined as a behavior, not an aspiration. Adapted, with the definitions kept close to the originals:

Discipline — being able to follow as well as to lead, knowing that we must master ourselves before we can master the task.
Competence — there being no substitute for total preparation, for the work will not tolerate the careless or indifferent.
Confidence — believing in ourselves as well as others, knowing that we must master hesitation before we can succeed.
Responsibility — realizing that it cannot be shifted to others, for it belongs to each of us; we must answer for what we do, or fail to do.
Persistence — taking a stand when we must, and trying again, even if it means following a more difficult path.
Teamwork — respecting and using the abilities of others, knowing that success depends on the efforts of all.
Vigilance — ★ never accepting success as a substitute for rigor in everything we do.

To which this handbook adds two that no operational profession needs, because operational professions are given their objectives:

Curiosity — treating the thing that does not fit as the most interesting object in the room, every time, including when it is inconvenient, including when it is the fourth one this week.
Honesty about uncertainty — saying what you do not know, in the paper and in the meeting, at the moment it is least comfortable to say it.
Appendix F — Reading
Small teams, reliability, and operations

Weick & Sutcliffe, Managing the Unexpected — high-reliability organizations; the most rigorous treatment of everything in this handbook
Beyer et al., Site Reliability Engineering and The SRE Workbook — free online; incident response, blameless review, error budgets, toil
Kranz, Failure Is Not an Option — mission operations culture from the inside
Bungay, The Art of Action — leading through intent, translated to organizations
Edmondson, The Fearless Organization — the evidence base for Chapter 32
Vaughan, The Challenger Launch Decision — how good people normalize deviance

Judgment and error

Klein, Sources of Power — how experts actually decide
Kahneman, Thinking, Fast and Slow
Croskerry's work on clinical debiasing — the source literature for Chapter 46
Gawande, The Checklist Manifesto
Petroski, To Engineer Is Human — failure as the engine of engineering knowledge
Klein & Roodman, "Blind Analysis in Nuclear and Particle Physics" — the method in Chapter 18

Writing

Gopen & Swan, "The Science of Scientific Writing" (American Scientist, 1990) — free, ten pages, the highest-return thing in this list
Williams, Style: Lessons in Clarity and Grace — the deep version of the same argument
Schimel, Writing Science — structure and story in scientific papers
Whitesides, "Whitesides' Group: Writing a Paper" — three pages on the outline-first method
Mensh & Kording, "Ten Simple Rules for Structuring Papers"
Pinker, The Sense of Style — the curse of knowledge
Boice, Professors as Writers — the evidence for brief daily sessions over binges

Doing science

Hamming, The Art of Doing Science and Engineering — read "You and Your Research" annually
Medawar, Advice to a Young Scientist
Polanyi, The Tacit Dimension — why apprenticeship is irreplaceable
Wilson et al., "Good Enough Practices in Scientific Computing" — the practical floor
Nosek et al. on preregistration and the exploratory/confirmatory distinction

Automation, and what it does to people

Bainbridge, "Ironies of Automation" (1983) — six pages, still the best thing written on the subject
Endsley on situation awareness and automation-induced complacency
Parasuraman & Manzey on complacency and bias in automated systems
The evidence base on skill acquisition under assistance is thin and moving; treat Chapter 47 as provisional and revisit it

Aviation and air traffic

Flight Safety Foundation ALAR Briefing Notes — free; stabilized approach criteria, go-around, monitoring
FAA Order JO 7210.3, Chapter 17 — Monitor Alert Parameter, ground delay programs, flow management
SKYbrary (skybrary.aero) — the best free reference on crew practice, TEM, and sterile cockpit
Gladwell's chapter on cockpit culture in Outliers is a readable entry point; Helmreich's CRM papers are the real literature

Craft and practice

Sennett, The Craftsman
Pye, The Nature and Art of Workmanship — workmanship of risk versus of certainty; the theoretical basis of Chapters 42 and 45
Alexander, The Timeless Way of Building — where the idea of a pattern language came from before software borrowed it
Isaacson, Steve Jobs — for the fence and the inside of the machine, if only for the anecdote
Ericsson on deliberate practice
Liker, The Toyota Way — the source of stop-the-line and continuous improvement



Revise after every review. If a chapter has not changed in a year, either it is correct or nobody is reading it, and the second is more likely.

