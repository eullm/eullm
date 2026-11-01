# Audit degli esperimenti Legal-IT 8B/9B (1 novembre 2026)

Ricognizione di quanto è stato provato per migliorare il ragionamento giuridico
italiano dei modelli legal-it, su Leonardo (EHPC-AIF-2026PG01-1147) e JUPITER
(EHPC-AIF-2026PG01-1434). Nessun training è stato avviato per scriverlo.

**Fonti.** Ogni affermazione rimanda a una di queste, sul ramo `main` salvo
dove indicato:

| Sigla | Documento |
|---|---|
| EXP | `docs/legal-it-experiments.md` (§1-§13, registro di Leonardo) |
| DOS | `docs/jupiter/dossier.md` (registro di JUPITER, § e data) |
| MC8 / MC4 | `forge/model_cards/legal-it-8b/README.md`, `legal-it-4b/README.md` |
| STR | `docs/legal-it-4b-strategy.md`, `docs/adr-001-offline-distillation.md` |
| MEM | `docs/research/legal-memory-plan.md` (ramo `feat/leonardo-legal-it`) |
| FB | `docs/feedback/legal-it.md` |

I job, le risposte e gli insiemi privati (LexIA, esame held-out, libro chiuso)
stanno su `$WORK` di Leonardo, sul `$PROJECT` di JUPITER e nell'archivio sulla
workstation: qui solo totali, come le regole di quegli insiemi chiedono.

## 0. Che cosa misurano i benchmark (vale per tutto il resto)

| Benchmark | Dimensione | Cosa misura davvero | Chi scrive / chi valuta | Fonte |
|---|---|---|---|---|
| Esami riservati del model card | 429 | domande costruite **automaticamente dal testo di un articolo**: cosa prevede, quale termine fissa (per numero o per argomento), articoli inesistenti. **Libro aperto**: 3 testi recuperati nel prompt, con la nota se l'articolo manca. 219 su 429 verificabili senza giudice | programma / giudice Qwen3-30B-A3B-Instruct-2507, rubrica | MC8 "Risultati" |
| devbig (sviluppo) | 472 | come sopra; con e senza nota | idem; 190 anche da programma | EXP "The exams" |
| Esame held-out di Leonardo | 233 | come sopra, chiesto una volta | idem | EXP §13 |
| Esame held-out di JUPITER | 217 | contenuto, termini, articoli inesistenti; da Normattiva | idem | DOS §5.8 |
| Esame pratico | 145 (+70 held-out) | **una domanda pratica che richiede un gruppo di articoli collegati**: l'unico che si avvicina all'applicazione a un caso | scritto da Qwen3.6-27B / giudice | EXP "The exams", §9 |
| Trappole LexIA (esterne) | 36 (33 senza le fuori ambito) | premesse false, articoli e sentenze inesistenti, fuori ambito | scritte da persone / giudice | EXP §7-§8 |
| Trap test generato + riscritto | 390 / 1.050 | gli stessi tipi di premessa, dal generatore | programma + giudice | DOS §5.11 |
| Esame giurisprudenza | 1.200 | rispondere da sentenze del Consiglio di Stato citando quella giusta | | EXP §3 |
| Libro chiuso | 716 | conoscenza senza testi: esistenza, numero ↔ rubrica, termini, pene, procedibilità | programma | EXP §13 |

**L'89,0% (e l'80,9 → 84,6 → 89,0) misura lettura e recupero su domande normative a
risposta puntuale, a libro aperto.** Non misura il ragionamento giuridico né
l'applicazione di norme a fatti: le domande nascono da un articolo solo, e la
risposta è nel testo che il modello ha davanti. L'unica misura vicina
all'applicazione è l'esame pratico (145), scritto e valutato da modelli.

**Accordo giudice-umano.** Il model card riporta che su 40 risposte valutate
alla cieca anche in modo indipendente l'accordo è stato tra il 68% e l'85% (MC8).
Il repository non dice chi abbia fatto la seconda valutazione, né su quale
esame: ❓. Sull'esame pratico non c'è alcuna misura di accordo. Il giudice è
quasi deterministico (cambia voto nello 0,6% dei casi su devbig, 3,3% sul pratico:
EXP §10), ma ha sbagliato in modo sistematico più volte (EXP §10-§12, `forge/CLAUDE.md`).

## A. Matrice delle tecniche

Legenda: ✅ testato con risultati · 🟡 parziale o inconcludente · ❌ non testato ·
❓ non verificabile. "8B" = Ministral-3-8B-Instruct-2512 e derivati; "9B/4B
Qwen3.5" = Qwen3.5-9B/4B-Base; i punteggi sono risposte corrette.

### A.1 Dataset e qualità dei dati

| Tecnica | Stato | Modello | Dataset | Prima → dopo | Evidenza |
|---|---|---|---|---|---|
| Corpus normativo pulito e deduplicato | ✅ | tutti | 9 codici Normattiva (Leonardo); norms-v2 = codici + 17.529 atti UE, 392.538 chunk, ~220M token (JUPITER) | n/a (è un dato) | DOS §5.9; `prepare_legislation.py`, `prepare_eurlex.py` |
| Dataset sentenze Cassazione | ✅ | | ItalGiure 2021-2026, ~1,13M chunk, ~700M token, pseudonimizzato (anonymize round 1-6), dedup; 325k sentenze civ+pen; CdS 48.673 | n/a | EXP §1; STR |
| Separazione fatti / motivazione / principio / dispositivo | 🟡 | teacher Qwen3-30B Q8 | schede: estratti principi, norme, esito, materia, domande; **fatti e dispositivo non separati come campi** | n/a | `eullm_forge/caselaw/schede.py`, `cds_schede.py` |
| Estrazione di questioni giuridiche dalle sentenze | 🟡 | idem | `domande_ricerca` e `domande_esame` per scheda (~230k schede); usate come prompt OPD, non valutate come qualità | n/a | `schede.py`; EXP §3 |
| Q/A con fonti verificabili | ✅ | 8B, 4B, Qwen3.5 | openbook-v04: 18.972 coppie (85% da un articolo, 15% articolo inesistente); v06: numero del passo + citazione letterale su 16.116; JUPITER 14.209 | 8B: 80,9 → 84,6 (solo SFT) sul 429 | MC8; EXP §2, §8; DOS §5.4 |
| Ragionamenti intermedi | 🟡 | 8B (r2b…), Qwen3.5 | formato "a tre passi controllati" delle premesse false (passo citato, verifica, risposta); nessuna catena di ragionamento giuridico generale | trappole 14 → 15-16 (nel rumore) | EXP §8-§9 |
| Applicazione di norme a casi concreti | 🟡 | 8B, 4B, Qwen3.5 | prompt pratici scritti da Qwen3.6-27B, usati solo come prompt GRPO (giudicati), mai come dati SFT verificati | pratico 60 → 87 (r2l) | EXP §9, §11 |
| Eccezioni, deroghe, conflitti normativi | ❌ | | nessun dataset dedicato | | — |
| Confronto fra sentenze analoghe e differenti | ❌ | | | | — |
| Controfattuali (cambia un fatto, cambia la conclusione) | 🟡 | 8B (r2d) | solo "gemelli a premessa vera" delle premesse false (stessa domanda con premessa giusta); non fatti di un caso | r2d 14 trappole (= pubblicato) | EXP §9 |
| Hard negatives (fonti pertinenti ma non applicabili) | 🟡 | tutti | i testi recuperati nelle coppie open-book sono distrattori naturali (BM25 + embedding + reranker); coppie con articolo tolto (absent-context); **nessun distrattore costruito apposta** (numero vicino, stesso numero di altro codice) | | MC8 "SFT"; EXP §7 |
| Mining degli errori dello student | ✅ | 8B, 4B | `sample_answers.py` (8 campioni), domande "miste" per GRPO, 158 coppie DPO, 249 esercizi di correzione | premesse: pass rate inesistente 0,95, abrogato 0,31, termine 0,40 | EXP §8 |
| Filtraggio automatico delle risposte errate del teacher | ✅ | | filtri del generatore (riferimenti "al testo", caso singolo, JSON); controllo programmatico delle riscritture (articoli, numeri, termini) | scarti 3-4% | DOS §5.4; MEM "Results" |
| Validazione umana o giuridica di un campione | 🟡 | 8B | 40 risposte (MC8, chi le ha valutate ❓); 16 risposte lette da un team legal-tech esterno; letture dell'utente su LexIA e sul corpus | | MC8; EXP §7; MEM |
| Controllo di contaminazione training/benchmark | ✅ | tutti | articoli degli esami esclusi da SFT e GRPO; JUPITER: 251 coppie su 9.223 su articoli d'esame (105 dei 153), rifatto sulle 8.972 pulite; `filter_grpo_prompts.py`; `check_corpus_holdout.py` | spostamenti di pochi punti | DOS §5.8, §5.11; EXP §9 |
| Split per sentenza, questione, periodo | ✅ | | per id di sentenza (`cds_split.py`); CdS 2025-26 e Cassazione 2025-26 solo valutazione; articoli d'esame per articolo | | EXP §1; DOS §7 |

### A.2 Pretraining e distillazione

| Tecnica | Stato | Modello | Dataset | Prima → dopo | Evidenza |
|---|---|---|---|---|---|
| CPT/DAPT su corpus normativo | ✅ | Qwen3 4B/8B, Qwen3.5 4B/9B (JUPITER) | norms-v2, 139M token, 1 epoca (via distillazione, non CE pura) | loss held-out 0,914 → 0,674 (4B), 0,827 → 0,625 (8B); esame: vedi sotto | DOS §5.8-5.9 |
| CPT su sentenze | ✅ | teacher Qwen3-30B-A3B-Base (fase 1) | corpus Cassazione 700M token | perplessità; nessun esame a domande | STR fase 1; EXP §1 |
| Distillazione KL teacher → student (logit) | ✅ | Qwen3-4B-Base (sett.), Qwen3/Qwen3.5 (JUPITER) | Cassazione (sett.); norms-v2 | sett.: perplessità 6,65 → 3,19, ma lettura peggiore (44 vs 67 su 86); JUPITER: vedi sotto | EXP §1; DOS §5.8 |
| Logit distillation con temperatura variabile | ❌ | | temperatura fissa 2,0, α 0,7 (o schedule 0,9 → 0,5 in STR); nessun confronto | | `distill.py` (`kl_temperature`); STR |
| Sequence-level KD | 🟡 | 8B, 4B | le coppie SFT scritte dal teacher sono di fatto KD a livello di sequenza; mai confrontata con la KD su logit a parità di dati | 80,9 → 84,6 | MC8 |
| Distillazione su risposte finali | ✅ | come sopra | | | MC8 |
| Distillazione di spiegazioni e ragionamenti intermedi | ❌ | | il teacher scrive risposte brevi; ragionamento del teacher mai distillato | | — |
| Distillazione IRAC | ❌ | | | | — |
| Distillazione selettiva delle sole risposte corrette del teacher | 🟡 | | filtri di formato, non di correttezza giuridica; le riscritture del corpus memoria hanno un controllo programmatico (numeri, articoli) | | DOS §5.4; MEM |
| Distillazione guidata dagli errori dello student | ❌ | | gli errori sono stati usati per GRPO e correzione, non per distillare dal teacher | | — |
| Feature / hidden-state distillation | ❌ | | | | — |
| Curriculum distillation | ❌ | | | | — |
| On-policy distillation a contesto privilegiato (reverse KL) | ✅ | Qwen3.5-4B/9B da Qwen3.8-27B; Ministral da se stesso | domande sulle sentenze CdS, il teacher vede la sentenza | devbig 9B 403 (sotto il pubblicato 437); giurisprudenza 0,348-0,362 per tutti, limitata dal retrieval (0,382) | EXP §3 |
| Confronto con teacher alternativi | ✅ | teacher candidati | Qwen3.6-27B, Mistral Small 4, Gemma 4 31B, Ministral Reasoning | trappole 16 / 10-12 / 13 / 4; pratico 76 / 55-60 / 74; nessuno sopra Qwen3.6-27B | EXP §8-§9; DOS §5.10 |
| Distillazione vs SFT a parità di dati e GPU-ore | 🟡 | Qwen3 4B/8B, Qwen3.5 | distillato vs Base sotto la stessa ricetta open-book; GPU-ore non pareggiate | 8B +9 senza nota; 9B Qwen3.5 +6/+8 | DOS §5.8 |
| Suite a libro chiuso (E0 del piano memoria) | ✅ | 8B pubblicato, r2l, r2q, jpn14, jpn17 | 716 domande dalla legislazione, valutate da programma | giuste 3-31%, 360-489 risposte sbagliate e sicure; il 30% di r2l è un riflesso ("non esiste") | EXP §13; MEM "Results" |
| CPT aumentato (molte forme per fatto) | 🟡 dati pronti, nessun training | backbone previsto r2l (e jpn17) | corpus memoria: 14.119 righe di programma, ~118k riscritture degli articoli in 5 varianti, ~1,85M righe di massime (civile e CdS); casi e spiegazioni esclusi dopo la lettura | — | MEM "Memory corpus"; `law_corpus.py`, `principle_rewrites.py` |
| Modulo di memoria n-gram hashed / product-key (stile Engram, Qwen3.8-Flash-Next) | ❌ | | | | MEM "Architecture" (solo progetto) |
| Memoria aggiornabile da sola (riforma simulata) | ❌ | | | | MEM H3 / E3 |
| Lookup della legge appreso con GRPO (tool) | ❌ | | | | MEM H4 / E5 |

**Verifica dei risultati riportati (KL 139M token):**

* **Confermati, con una precisazione.** 4B 0,684 → 0,689, 8B 0,747 → 0,744 sono il
  punteggio dell'esame JUPITER (corrette + metà delle parziali, su 217), dopo
  un tuning sulle **8.972 coppie di istruzioni pulite**, il 44% delle quali a
  libro chiuso (DOS §5.8, "Two more questions"). Non sono un'accuratezza pura.
* **Loss giù, accuratezza ferma: vero in quel setup**, falso detto in generale.
  Sotto la ricetta open-book il distillato è avanti in tutti e quattro i
  confronti: 8B +3 con nota e **+9 senza nota** (174 → 183), 4B +2/+1
  (job 2455477). Qwen3.5-9B +6/+8, 4B +2/+5 a fine epoca (job 2480052).
  L'esame è però vicino al tetto: cinque modelli entro 189/217.
* **Un costo non riportato.** Continuare la distillazione sul resto di
  norms-v2 (~81M token, run "dqr") ha fatto scendere il 9B sull'held-out da
  188 a 168 (DOS 26-10). Il GRPO lo ha riportato a 198.

### A.3 Supervised Fine-Tuning

| Tecnica | Stato | Modello | Dataset | Prima → dopo | Evidenza |
|---|---|---|---|---|---|
| SFT su sole norme | ✅ | 8B, 4B | openbook-v04 | 8B 80,9 → 84,6 | MC8 |
| SFT su sentenze | ✅ | 8B (dal pubblicato) | 75.352 coppie giurisprudenza + 20.572 norme, 6.007 passi | **peggiore**: devbig 437 → 415, pratico 60+21 → 57+9, trappole 14 → 4 | EXP §8 |
| SFT su Q/A giuridiche (closed-book) | ✅ | Qwen3/Qwen3.5 (JUPITER) | 8.972 coppie, 44% a memoria | sotto l'open-book di 7,8-15,5 punti | DOS §5.8 |
| SFT open-book | ✅ | tutti | 14.209 / 18.972 coppie | Qwen3.5-4B 0,786 → 0,864; 9B → 0,876 | DOS §5.8 |
| SFT senza documenti | ✅ | come sopra | | a libro chiuso tutti 0,000-0,023 | DOS §5.8 |
| SFT misto open/closed-book | ✅ | Qwen3/Qwen3.5 | clean + open-book | 4B 0,878 (meglio); 9B 0,820 (peggio di 0,876); 8B 0,836 (peggio di 0,866) | DOS §5.8 |
| RAFT (retrieval nel prompt di training) | ✅ | 8B, 4B | coppie open-book con i 3 testi del retrieval | come sopra | MC8 "Addestramento" |
| RAFT con irrilevanti / hard negatives | 🟡 | | distrattori naturali + coppie con articolo tolto; nessun hard negative costruito | | EXP §7 |
| SFT con ragionamento strutturato IRAC | ❌ (🟡 per le premesse) | 8B | formato a tre passi (r2b-steps vs plain) solo per le premesse false | trappole 15 vs 14 (nel rumore) | EXP §8-§9 |
| SFT su correzione di ragionamenti sbagliati | 🟡 | 8B (r2b) | 249 esercizi di correzione nel mix breve; effetto mai isolato | | EXP §8 |
| SFT con casi controfattuali | ❌ | | | | — |
| LoRA rank 16/32/64/128 | ❌ (confronto) | | r32 per SFT e GRPO, r128 per la distillazione; nessun confronto fra rank | | MC8; DOS §5.2 |
| Full fine-tuning vs LoRA | ❌ | | | | — |
| Learning rate ed epoche | 🟡 | Qwen3.5, Qwen3 | 1 vs 2 epoche: 4B 0,880 vs 0,864, 9B 0,862 vs 0,876 (nel rumore); lr non esplorato (1e-4 SFT, 1e-5 GRPO) | | DOS §5.8 |
| Controllo del catastrophic forgetting | 🟡 | 8B | osservato (SFT lungo dopo GRPO: EXP §8; dqr: DOS 26-10); **nessun test delle capacità generali** | | EXP §8 |
| Loss solo sulla risposta | ✅ (implementato, non confrontato) | tutti | prompt e padding mascherati dalla loss | nessun confronto con loss piena | `stage3_sft.py` docstring |

**Verifica dei risultati riportati:**

* **Qwen3.5-4B "open-book + closed-context" 87,8%: confermato, nome da
  correggere.** È 0,878 con le coppie open-book **più le 8.972 coppie di
  istruzioni** (il 44% a memoria), 2 epoche (DOS §5.8, tabella del 19-10).
  "Closed-context" va letto "istruzioni anche a libro chiuso".
* **9B open-book-only 87,6%: confermato** (0,876). Con il mix 0,820.
* **Il mix penalizza il 9B: confermato su un solo run**, senza semi diversi,
  su 217 domande (12 domande). Con la ricetta a 1 epoca il 9B open-book fa
  0,862: la variazione fra due run dello stesso tipo è vicina a quella del mix.
* **251 coppie contaminate: confermato.** Sono 251 su 9.223 coppie che
  chiedono di un articolo d'esame, 105 articoli su 153. Rifatto il tuning
  sulle 8.972 pulite: i punteggi si spostano di pochi punti, i Qwen3.5 restano
  davanti al riferimento (+9, +12) (DOS §5.8).

### A.4 Preference optimization e reinforcement learning

| Tecnica | Stato | Modello | Dataset | Prima → dopo | Evidenza |
|---|---|---|---|---|---|
| DPO | ❌ | | 158 coppie costruite (`make_correction_pairs.py --out-dpo`), **mai addestrate** | | EXP §8 |
| SimPO / ORPO | ❌ | | | | — |
| GRPO | ✅ | 8B (r2a…r2r), Qwen3.5 4B/9B (jp…, jpn…, g9…, g4…) | prompt su termini, inesistenti, premesse, pratici | vedi sotto | EXP §2-§13; DOS §5.11 |
| GRPO con reward sulla risposta finale | ✅ | | reward programmatiche (termine, astensione, premessa) | 84,6 → 89,0 (429) | MC8 |
| GRPO con reward sulla correttezza giuridica (giudice) | ✅ | 8B, Qwen3.5 | prompt pratici e "contenuto" valutati dal giudice | pratico 60 → 87 (r2l); 9B 67 → 81 (g9r-r2) | EXP §9-§11; DOS 27-10 |
| GRPO con reward sulla fedeltà alle fonti | 🟡 | Qwen3.5 (c3) | `--citation-weight 0.3` | nessun effetto (circa il 90% già citava l'articolo giusto); abbandonato | DOS 29/30-10 |
| GRPO con reward sull'applicazione ai fatti | ❌ | | il giudice dà un voto unico alla risposta pratica; nessuna reward per requisito o per fatto | | — |
| Process Reward Model / process supervision | ❌ | | | | — |
| Reward per eccezioni e controargomenti | ❌ | | | | — |
| Reward per coerenza conclusione-motivazione | ❌ | | | | — |
| Preferenze da risposte corrette/errate dello student | 🟡 | | coppie costruite, non addestrate; le stesse domande usate per GRPO | | EXP §8 |
| Penalizzazione di citazioni inventate o tagliate | ✅ | 8B (r2l) | penalità per citazione tagliata; reward di citazione (sopra) | citazioni tagliate 16 → 8 su 472 | EXP §11 |
| Verifica indipendente dei reward (anti reward hacking) | ✅ | | ogni reward controllato su risposte vere in tutte e due le direzioni; 5 reward sbagliati trovati leggendo le risposte | | EXP §8-§12; `forge/CLAUDE.md` |
| Generalizzazione dopo RL su casi nuovi | 🟡 | 8B, 4B | esame held-out chiesto una volta | r2l 173/174 contro 168/169 del pubblicato (+5, p 0,41-0,44, **non significativo**); il guadagno resta sul pratico | EXP §13 |

**Verifica 80,9 → 84,6 → 89,0 (429 domande): confermato** (MC8). Misura
domande puntuali a libro aperto (§0), non ragionamento complesso. Dentro ci
sono domande sugli articoli inesistenti e sui termini; non ci sono casi con
più fatti o più norme. Il GRPO del modello pubblicato premiava termini e
astensione, quindi il +4,4 si concentra lì (219 domande verificabili senza
giudice: 199 → 202).

Ore GPU documentate (EXP §2): 8B SFT ~4,2 GPU-h, GRPO ~4,6 GPU-h; coppie del
teacher 40,2; esami e giudice 87,7 condivisi. Per i round r2* successivi le ore
del singolo round non sono documentate (❓); un round su JUPITER (g9-r1) costa
circa 2 ore di nodo, 8 GPU-ore (DOS 26-10).

### A.5 Ragionamento giuridico

| Capacità | Stato | Evidenza |
|---|---|---|
| Identificazione della questione | ❌ | nessuna misura dedicata |
| Fatti decisivi | ❌ | |
| Presupposti normativi | ❌ | |
| Applicazione separata di ogni presupposto | ❌ | il pratico ha una rubrica unica; la "rubrica per parte" è proposta (EXP §7) ma non risulta eseguita |
| Eccezioni | ❌ | |
| Confronto fra orientamenti giurisprudenziali | ❌ | |
| Ratio decidendi vs obiter | ❌ | |
| Precedente applicabile vs non pertinente | 🟡 | `caselaw_wrong_cite` nella Cassazione SFT (EXP §7-§8), modello peggiorato; non misurato a parte |
| Norme modificate o abrogate | ✅ | premesse "abrogato" (pass 0,31 → saturato), libro chiuso "abrogato"; LexIA abuso d'ufficio: "abrogato" senza conseguenza (EXP §13) |
| Normativa vigente alla data del fatto | ❌ | una sola trappola LexIA (abuso d'ufficio 2024); nessun dataset |
| Conclusioni condizionate a fatti mancanti | ❌ | |
| Riconoscimento dell'insufficienza delle fonti | ✅ | absent-context, nota/no nota, context gate (scartato: fermava 31-33 trappole su 36), off-topic (chiuso: insegnava una frase) — EXP §10-§12 |
| Multi-hop con più articoli | 🟡 | esame pratico (gruppi di articoli): metà delle risposte pratiche sbagliate cita un articolo fuori dal gruppo (DOS 27-10, `citation_check.py`) |
| Self-correction | 🟡 | esercizi di correzione nel mix r2b, non isolati |
| Risposta diretta vs ragionamento strutturato | 🟡 | tre passi vs paragrafo per le premesse (r2b-steps vs plain): nel rumore |
| Più token di ragionamento → più accuratezza | 🟡 | **solo sui teacher**: Qwen3.6-27B con reasoning peggio sulle trappole (16 → 10), meglio sul pratico (76 → 85); Ministral Reasoning 6 → 4 trappole; Mistral Small 4 10 → 12. **Mai sugli student**: il reasoning di Qwen3.5 resta spento (EXP §3, §8-§9) |

### A.6 Benchmark e valutazione

| Voce | Stato | Evidenza |
|---|---|---|
| "OPD-v2", "OPD-v3" | ❓ | nel repository non esistono benchmark con questo nome; OPD è il training a contesto privilegiato (EXP §3) |
| Benchmark Cassazione | 🟡 | esame giurisprudenza (1.200) sul Consiglio di Stato; nessun esame sulla Cassazione documentato |
| Questioni normative separate | ✅ | 429 / devbig / held-out |
| Casi giurisprudenziali separati | 🟡 | esame CdS, limitato dal retrieval |
| Applicazione dei fatti | 🟡 | esame pratico, scritto e valutato da modelli |
| Accuratezza della conclusione | ❌ | |
| Coerenza controfattuale | ❌ | |
| Gestione delle eccezioni | ❌ | |
| Verifica citazioni e fonti | ✅ | `citation_check.py`, `check_cited_articles.py`, citazioni tagliate |
| LLM judge | ✅ | Qwen3-30B-A3B-Instruct-2507, rubrica v3 |
| Accordo giudice-umani | 🟡 / ❓ | 40 risposte 68-85% (chi ❓); nessuna misura sul pratico |
| Valutazione blind con fonti identiche | ✅ | stesso retrieval, stessi prompt per tutti i modelli; McNemar appaiato |
| Fonti corrette già fornite, senza retrieval | ❌ | ogni esame passa dal retrieval |
| Fonti corrette + distrattori | ❌ (controllato) | i distrattori sono quelli del retrieval, non scelti |
| Generalizzazione su nuove sentenze | ❌ | CdS 2025-26 e Cassazione 2025-26 tenute fuori ma mai usate come esame di ragionamento |
| Regression test sulle capacità generali | ❌ | |

### A.7 Diagnosi dei limiti

| Voce | Stato | Evidenza |
|---|---|---|
| Stessi casi student vs teacher | ✅ | pratico: r2l 87, Qwen3.6-27B 76 (85 con reasoning), Qwen3-30B 74 (EXP §9, §11) |
| Modello di ragionamento più forte | 🟡 | 27B reasoning, Mistral Small 4 reasoning; nessun modello frontier sugli stessi set (la piattaforma esterna su 5 domande: EXP §13) |
| Casi risolti dal teacher e sbagliati dallo student | 🟡 | per trappola su 7 modelli (EXP §9); non sistematico sul pratico |
| Casi sbagliati da tutti | ✅ | 11 trappole su 36 sbagliate da tutti, in tre tipi (EXP §9) |
| Errori per categoria giuridica | 🟡 | per tipo di premessa e di trappola; non per materia |
| Errori per complessità e lunghezza del contesto | ❌ | solo il report esterno su prompt da 50-80k token (EXP §7) |
| Sensibilità all'ordine dei documenti | ❌ | |
| Conclusioni contraddittorie | ❌ | segnalate dal report esterno, non misurate |
| Quantizzazione Q4/Q8/FP16 | ✅ | bf16 435 vs Q4 437 (MC8); Qwen3.5 Q4 costa 0-6 domande (DOS §5.8, §5.11) |
| Base vs SFT vs GRPO vs OPD | ✅ | MC8; EXP §3; DOS §5.11 |
| Limiti di capacità dell'architettura 8/9B | 🟡 | 14B = 8B su devbig (436 vs 437, EXP §2); libro chiuso 3-31% (EXP §13); nessuna curva per scala sul pratico |

## B. Esperimenti falliti o inconcludenti

**1. Distillazione su Cassazione (settembre, 4B).**
- *Obiettivo:* un 4B che conosce il diritto attraverso le sentenze.
- *Osservato:* perplessità 6,65 → 3,19; 4 risposte su 10 sbagliate lette da una
  persona; nella lettura di un articolo peggio dell'Instruct (44 vs 67 su 86);
  18 articoli inesistenti su 18 inventati (EXP §1).
- *Perché:* **dimostrato** che la perplessità non predice la lettura. **Ipotesi:**
  il testo grezzo delle sentenze insegna lo stile e non i fatti recuperabili;
  senza SFT il modello non risponde a domande.
- *Riprovare?* Solo come "distillato + ricetta open-book", fatto poi su JUPITER
  con le norme (+9 senza nota per l'8B Qwen3). Sulle sentenze mai.

**2. Distillazione sulle norme, 139M token (JUPITER).**
- *Obiettivo:* conoscenza delle norme.
- *Osservato:* loss −24/−26%; esame open-book fermo con le coppie di istruzioni;
  +1/+9 con la ricetta open-book; il proseguimento su altri 81M token costa 20
  domande al 9B.
- *Perché:* **dimostrato** che a libro aperto la norma è nel prompt, quindi
  conoscerla aggiunge poco; a libro chiuso tutti restano a 0-2%. **Ipotesi:**
  una passata di testo non rende i fatti richiamabili (MEM, "Why the knowledge
  did not go in").
- *Riprovare?* Sì, ma come continued pretraining **aumentato** (molte forme per
  fatto), misurato sul libro chiuso: è E1 del piano memoria, con il corpus già
  generato su Leonardo.

**3. SFT lungo sulla giurisprudenza sopra il modello pubblicato.**
- *Osservato:* devbig −22, pratico peggio, trappole 14 → 4; tutte le risposte
  LexIA uguali parola per parola.
- *Perché:* **dimostrato** dalle risposte. Il 14% del mix erano rifiuti a frase
  fissa, e l'SFT lungo disfa il GRPO.
- *Riprovare?* Solo con SFT corto (rifiuti sotto il 6%) seguito da GRPO, e con
  le sentenze nel prompt; ma l'esame giurisprudenza è limitato dal retrieval
  (0,382), quindi prima il retrieval.

**4. OPD a contesto privilegiato (Qwen3.5 da Qwen3.8-27B).**
- *Osservato:* partenza sotto il pubblicato (403 vs 437); giurisprudenza
  0,348-0,362 per ogni round.
- *Perché:* **dimostrato** che il tetto è il retrieval (sentenza giusta trovata
  nel 38% dei casi, citata 9 volte su 10 quando c'è). **Ipotesi:** OPD insegna
  la forma della risposta, non quale sentenza cercare.
- *Riprovare?* No, finché il retrieval delle sentenze non migliora.

**5. Round GRPO sulle premesse generate (r2d … r2r, jpn …, g9-r3, g4-r4/r5).**
- *Osservato:* saturazione (gruppi tutti uguali, zero-std 0,67-1,00); r2o, r2q,
  jpn19, g9r-r3 pari ai precedenti nel rumore; l'held-out +5 non significativo.
- *Perché:* **dimostrato** dai log: niente da imparare dove il modello risponde
  sempre giusto o sempre sbagliato. Il pratico è salito una volta (60 → 87) e si
  è fermato.
- *Riprovare?* No con gli stessi prompt. Servono prompt nuovi, più difficili, con
  una misura capace di distinguerli.

**6. Prompt fuori ambito con reward di astensione (r2m, r2n, jpn15/16).**
- *Osservato:* trappole 14 → 10 → 7; il modello ha imparato una frase.
- *Perché:* **dimostrato:** il reward premiava qualunque astensione, e
  `premessa_inesistente` era anch'esso "astensione" (bug corretto, `denies_existence`).
- *Riprovare?* Chiuso. La versione giudicata (r2p, jpn18) non ha raggiunto le
  trappole; su quelle tre domande il giudice è una moneta.

**7. Context gate (riclassificatore 0,6B davanti al modello).**
- *Osservato:* 0,92-0,93 sul suo dev set; fermava 31-33 trappole su 36 e fino a
  77 domande pratiche su 145.
- *Perché:* **dimostrato:** addestrato su domande brevi a un articolo, leggeva
  premesse false e casi con più fatti come senza risposta.
- *Riprovare?* Solo con esempi del tipo delle trappole, che oggi non abbiamo
  senza toccare il set di prova.

**8. Reward di citazione (g9r-c3, g4-c3).**
- *Osservato:* invariato. Circa il 90% delle risposte cita già l'articolo della
  domanda; gli errori sono una scelta sbagliata fra i testi recuperati.
- *Riprovare?* Non in quella forma: serve una reward o dati che vedano la
  scelta fra distrattori (sezione C).

**9. Dati in più su altro testo (19.141 coppie UE, JUPITER).**
- *Osservato:* −3/−9 domande su ogni modello. **Dimostrato:** coppie sul testo
  dell'esame contano, altre coppie no. Da non ripetere.

**10. Teacher alternativi** (Mistral Small 4, Gemma 4 31B, Ministral Reasoning).
- *Osservato:* nessuno sopra Qwen3.6-27B (EXP §9, DOS §5.10). Il pratico di r2l
  (87) supera già i teacher senza reasoning e pareggia il 27B con reasoning (85).
- *Conseguenza:* distillare dai teacher disponibili non può insegnare il
  pratico oltre il livello attuale.

**11. Corpus memoria, forme che elaborano** (casi pratici e spiegazioni del
teacher, 30-10).
- *Osservato:* 3 sbagliati su 6 in ciascuna forma, letti a mano.
- *Perché:* **dimostrato** dalla lettura: il teacher applica la norma e la
  inquadra male (qualità invece di natura, art. 1429 c.c.; 540-bis su immobili).
- *Riprovare?* Non con questo teacher senza una verifica giuridica.

## C. Tecniche realmente mancanti

Solo quanto non eseguito o eseguito in modo non conclusivo. Costo GPU in ordine
di grandezza, dai costi documentati (SFT 8B ~4 GPU-h, round GRPO ~8-25 GPU-h,
esami e giudice 10-20 GPU-h a giro); nessuna stima di guadagno.

| # | Tecnica | Potenziale sul ragionamento | Costo GPU | Complessità | Rischio | Priorità |
|---|---|---|---|---|---|---|
| C1 | **Benchmark di applicazione**: casi con fatti, fonti date (giuste + distrattori scelti), rubrica per requisito, gemelli controfattuali, campione validato da un giurista | prerequisito: senza, nessuna delle voci sotto è misurabile | basso (generazione + giudice, < 50 GPU-h) | media (serve tempo di un giurista) | nessuno sul modello | **1** |
| C2 | Hard negatives costruiti (articolo vicino, stesso numero di altro codice, norma abrogata) in SFT corto e/o reward sulla scelta | alto sull'errore misurato (metà delle risposte pratiche sbagliate) | basso-medio (SFT corto + 1 round GRPO, ~30-50 GPU-h) | media | basso (SFT corto, poi GRPO) | **2** |
| C3 | Reward per requisito (rubrica scomposta) e per coerenza controfattuale nel GRPO pratico | alto se C1 lo misura; il GRPO pratico è l'unica tecnica che ha mosso il ragionamento (60 → 87) | medio (round GRPO, ~25-50 GPU-h; il giudice per requisito costa di più) | media-alta | medio (reward hacking sulla rubrica: va letta) | **3** |
| C4 | DPO / ORPO / SimPO sulle risposte proprie | basso-medio: lo stesso segnale del GRPO, più economico | basso | bassa (coppie pronte) | basso | 5 |
| C5 | Distillazione del ragionamento del teacher, solo risposte verificate, sui casi falliti dallo student | incerto: i teacher disponibili non superano r2l sul pratico | medio | media | medio | 6 |
| C6 | Reasoning acceso negli student Qwen3.5 (valutazione, poi eventualmente GRPO) | incerto: sui teacher aiuta il pratico e danneggia le trappole | basso per la sola valutazione | bassa | nessuno da valutare | 4 |
| C7 | Normativa vigente alla data del fatto (versioni storiche Normattiva) | medio sulle domande di diritto intertemporale | basso (dati) + SFT/GRPO | alta (dati da prendere) | basso | 7 |
| C8 | Eccezioni e deroghe, multi-norma, sensibilità all'ordine e alla lunghezza | medio | basso per misurare | media | nessuno | dentro C1 |
| C9 | Regression test sulle capacità generali | protegge, non migliora | basso | bassa | nessuno | dentro ogni round |
| C10 | LoRA rank, full FT, lr, curriculum, temperatura KD | basso: nessun segnale che il limite sia l'ottimizzazione | medio | bassa | basso | 8 |
| C11 | PRM / process supervision | alto in teoria, ma richiede passaggi annotati che non esistono | alto | alta | alto | dopo C1-C3 |
| C12 | Piano memoria E1: CPT aumentato su r2l, LoRA con replay del post-training | conoscenza senza testi (libro chiuso) e la legge fuori dall'indice (LexIA: conseguenza dell'art. 2 c.p., sospensione feriale) | 120 GPU-h a 2B token, 600 a 10B | media (corpus pronto) | medio (un training lungo disfa il GRPO: replay ed esami a libro aperto dopo) | **deciso**, sezione E |
| C13 | Piano memoria E2: modulo n-gram hashed in RAM, backbone congelato, tre taglie | conoscenza a parametri attivi uguali; backbone intatto per costruzione | 3 × 600 GPU-h | alta (modulo nuovo in Forge, operatore nel motore per E4) | medio-alto: i guadagni noti sono su modelli addestrati da zero | **deciso**, dopo E1 |
| C14 | Piano memoria E3-E5: memoria aggiornata da sola, operatore nel motore, lookup appreso | aggiornare la legge senza riaddestrare; consultare quando incerto | < 50 + ore di banco + 200-400 GPU-h | alta | medio | dopo E2 |

## D. Raccomandazione finale: tre esperimenti

L'obiettivo è applicare norme e giurisprudenza a casi concreti **con le fonti
nel contesto**. Oggi nessun benchmark lo misura in modo affidabile (§0). Per
il vincolo 6 il primo passo non è un training.

1. **C1: il benchmark di applicazione, prima di ogni nuovo training.**
   Circa 150-300 casi su articoli fuori da ogni training. Ogni caso ha:
   - i fatti e le fonti date in tre condizioni: giuste, giuste + distrattori
     scelti, mancanti;
   - una rubrica per requisito (questione, norma, ogni presupposto applicato,
     eccezione, conclusione);
   - un gemello controfattuale in cui cambia un fatto decisivo, e la
     conclusione deve cambiare.

   Un giurista valuta un campione, e si misura l'accordo del giudice per
   requisito. **Perché è un progresso:** l'esame pratico è l'unico vicino, ma
   ha voto unico, è scritto da un 27B e non ha misure di accordo. Senza C1 il
   plateau di r2l/g9r-r2 non si può distinguere dal rumore.
2. **C2: hard negatives costruiti, poi un round GRPO.** È l'errore più frequente
   misurato: metà delle risposte pratiche sbagliate cita un articolo fuori dal
   gruppo. La reward di citazione non lo ha toccato perché paga la presenza di
   una citazione, non la scelta fra testi simili. **Cosa cambia rispetto al già
   fatto:** distrattori scelti apposta invece di quelli casuali del retrieval.
   SFT corto (la regola di EXP §8), poi GRPO; misurato su C1 nella condizione
   "giuste + distrattori".
3. **C3: GRPO pratico con reward per requisito e coerenza controfattuale.** Il
   GRPO pratico è l'unica tecnica che ha mosso il ragionamento (60 → 87 sull'8B,
   67 → 81 sul 9B) e si è fermato perché il voto unico non distingue più le
   risposte. **Cosa cambia:** la reward paga ogni presupposto applicato e il
   cambio di conclusione sul gemello, con prompt nuovi più difficili presi dai
   casi che r2l sbaglia. Va letto in entrambe le direzioni prima di fidarsi, e
   misurato su C1.

Questi tre riguardano il ragionamento **con le fonti nel contesto**, la
domanda di questo audit. Il piano memoria (sezione E), deciso come prossimo
lavoro, riguarda la **conoscenza senza fonti**: è un binario parallelo e non
alternativo, con un proprio benchmark già costruito (il libro chiuso). C1 serve
anche a lui: un modello che sa di più deve ancora applicare bene quello che sa.
Distillare dai teacher disponibili non è prioritario, perché nessuno supera
r2l sul pratico.

## E. Il piano memoria (deciso il 26-10, `docs/research/legal-memory-plan.md`)

Obiettivo: un modello piccolo che **sa** la legge, non solo la legge, dato che
oggi la legge la legge soltanto. In concreto:
- corregge una premessa a libro chiuso;
- quando non è sicuro consulta la legge da sé;
- si aggiorna quando la legge cambia, senza riaddestrare il modello.

L'idea è quella delle tabelle n-gram hashed di Qwen3.8-Flash-Next e di
DeepSeek Engram: una tabella da 1-6 miliardi di parametri, letta a uno strato,
in RAM e precaricata, salvata in un file versionato (`lex-memory-YYYY-MM`).

| Esperimento | Ipotesi | Stato | Benchmark che lo misura | Costo (GPU-h) |
|---|---|---|---|---|
| E0 | punto di partenza | ✅ suite a libro chiuso (716) e misura dei modelli attuali; corpus memoria generato su Leonardo (sopra, A.2) | libro chiuso | < 20 (speso) |
| E1 | H1: il CPT aumentato fa ricordare la legge | ❌, dati pronti | libro chiuso (sì se ≥ +25 punti su r2l), senza perdite su devbig, pratico, trappole | 120 (2B token), 600 (10B) |
| E2 | H2: il modulo n-gram aggiunge ricordo a parametri attivi uguali | ❌ | E1 con e senza modulo, stessi dati e token (sì se ≥ +10 punti) | 3 × 600 |
| E3 | H3: la memoria si aggiorna da sola | ❌ | riforma simulata di 50 articoli (sì se ≥ 80% del nuovo testo, vecchio ricordo entro 2 punti) | < 50 |
| E4 | il lookup come operatore nel motore | ❌ | velocità su GPU e CPU (LUMI) | ore di banco |
| E5 | H4: il modello impara quando consultare la legge | ❌ | trappole su 33 ed esame pratico sopra il miglior modello a libro aperto | 200-400 |
| E6 | scala del vincitore | ❌ | tutti i precedenti | 2.000+ |

Cosa dice l'audit sul piano:
- **E1 è il primo go/no-go.** Se a 2 miliardi di token il CPT aumentato non muove
  il libro chiuso, il problema non è il modulo e il piano cambia prima di
  spendere su E2.
- **Il rischio maggiore è noto.** I guadagni dei moduli di memoria sono mostrati
  su modelli addestrati da zero; innestati su un modello già addestrato possono
  dare meno, ed E2 misura proprio questo. L'altro rischio è già misurato due
  volte: un training lungo disfa il GRPO (EXP §8; DOS 26-10, dqr). Per questo E1
  usa LoRA con replay e gli esami a libro aperto dopo, ed E2 tiene il backbone
  congelato.
- **Cosa resta da decidere prima di E1:** le massime penali (135k schede,
  escluse dal filtro attuale) e i numeri veri delle sentenze accanto ai
  principi.
- **Privacy:** il test di estrazione e i canary del piano valgono per ogni
  modello addestrato su testi derivati da sentenze, prima di un rilascio.
- **Il rapporto con C1-C3:** E1/E2 danno al modello la legge che non ha davanti,
  e C1-C3 gli insegnano ad applicare quella che ha davanti. Il prodotto ha
  bisogno di tutti e due; il libro chiuso misura il primo, C1 il secondo.
