# Roadmap Legal-IT: ragionamento con le fonti e memoria della legge

Da seguire su JUPITER da novembre 2026. Scritta il 1° novembre 2026, dopo la
chiusura di Leonardo. Viene da due documenti, da leggere prima di cominciare:
`docs/research/legal-it-audit-2026-11.md` (cosa è stato provato e con quale
esito) e `docs/research/legal-memory-plan.md` (il piano memoria E0-E6).

**Due obiettivi, misurati separatamente:**

* **Ragionamento (C):** con le fonti giuste davanti, il modello sceglie quella
  pertinente, la applica ai fatti e conclude bene. Si misura con il benchmark
  C1, da costruire.
* **Memoria (E):** senza fonti, il modello sa la legge. Si misura con il libro
  chiuso, 716 domande, `make_closed_book.py` e `score_closed_book.py`.

**Regola generale:** nessuna fase di training parte finché la sua condizione di
ingresso non è soddisfatta e il benchmark che ne misura il beneficio non
esiste. Ogni numero si decide solo dopo aver letto le risposte
(`compare_graded.py --show`). Ogni risultato va anche in
`docs/jupiter/dossier.md`.

## Fase 0: prerequisiti (poche GPU-ore, nessun training)

### 0.1 Portare su JUPITER solo quello che è permesso

Finché JSC non firma l'addendum sulla protezione dei dati (dossier §7) su
JUPITER va solo ciò che non deriva dal testo delle sentenze.

| Va su JUPITER | Resta fuori fino all'addendum |
|---|---|
| r2l fuso (`grpo-merged/r2l-steps-8b-quote`) e GGUF/adapter di r2l e r2q: discendono dalle norme e dalle schede pulite (`make_caselaw_sft.py`, solo principi astratti con riferimenti inventati) | jpn14, jpn17 e tutta la linea jp/jpn: discendono dall'OPD con passi di sentenze nel prompt |
| corpus memoria degli articoli: `program.jsonl`, `rewrites*.jsonl` | schede (`cds`, `cass`), `principi*.jsonl` (portano l'id della sentenza), `datasets/legal_it*` |
| libro chiuso, devbig, pratico, trappole (le ultime già su JUPITER per le run del 24-10) | esame giurisprudenza (sentenze del Consiglio di Stato) |
| premise6/premise7, prompt GRPO di Leonardo (già filtrati da `filter_grpo_prompts.py`) | checkpoint della distillazione di settembre (`legal_it*`) |

Dove sono le copie:
* server I3K (`~/eullm-archive/leonardo`, ~330 GB, verificato con prova a vuoto);
* workstation (`~/eullm-archive`, 88 GB, verificato);
* modelli fusi dello scratch, su un VPS temporaneo in UE.

Per trasferire dal server I3K a JUPITER serve una chiave registrata su JuDoor
con `from=` uguale all'indirizzo di quel server.

Le massime civili e del Consiglio di Stato si possono portare in una versione
**senza id** (tolti `card`, `req`, `id`; i soli principi riscritti, già
filtrati da `why_not`), cioè "ruling summaries that carry nothing of the case"
come il §7 ammette. È una decisione dell'utente, e serve un piccolo script di
esportazione: non esiste ancora.

### 0.2 Confronto r2l contro g9r-r2: decide il backbone 8/9B

Non sono mai stati misurati sugli stessi esami.
* **Modelli:** r2l (Ministral 8B, Leonardo) e g9r-r2 (Qwen3.5-9B, JUPITER);
  come riferimenti g4-r3 (4B) e il legal-it-8b pubblicato.
* **Set:** esame held-out JUPITER (217), devbig (472) con e senza nota, pratico
  (145), LexIA (36 e 33), libro chiuso (716), trap test (390) e la sua
  versione riscritta (1.050).
* **Condizioni:** stesso retrieval e stesso giudice per tutti; Q4_K_M
  attraverso EULLM Engine, come il prodotto.
* **Costo:** circa 10-20 GPU-ore.
* **Esito:** il backbone di E1 e di C2/C3. In caso di sostanziale parità
  conta lo spazio su una GPU da 16 GB e la licenza: entrambi Apache 2.0.

### 0.3 Allargare l'indice del retrieval a tutti gli atti in vigore

Su LexIA due errori su cinque venivano da leggi assenti dalla raccolta (la
l. 742/1969 della sospensione feriale, le conseguenze dell'art. 2 c.p.;
EXP §13).
* **Come:** `prepare_legislation.py --all-acts`, poi si rifanno devbig senza
  nota, pratico e LexIA su r2l e g9r-r2.
* **Cosa misura:** quanto di "non sa la legge" è in realtà "la legge non è
  nell'indice". Nessun training. Va fatto prima di attribuire quegli errori
  alla memoria.

### 0.4 Test di regressione sulle capacità generali

Oggi non esiste. Serve un piccolo set fisso: per esempio gsm8k e ARC, già
nella campagna `bench/campaign`, più un set di comprensione in italiano. Si
esegue su ogni checkpoint prodotto da qui in avanti.

## Fase 1: C1, benchmark di applicazione (ragionamento)

**Ingresso:** fase 0.2 fatta.

* **Casi:** 150-300, su gruppi di articoli collegati, **fuori da ogni dato di
  training** (stessa esclusione di `excluded()` e `filter_grpo_prompts.py`).
  Ogni caso ha:
  - fatti;
  - questione;
  - norme applicabili;
  - eccezioni;
  - conclusione;
  - rubrica per requisito (questione, fatti decisivi, ogni presupposto
    applicato, eccezione, coerenza tra motivazione e conclusione);
  - un gemello controfattuale con un fatto decisivo cambiato e la conclusione
    che cambia.
* **Tre condizioni, montate da programma:**
  1. solo le fonti giuste;
  2. fonti giuste più distrattori scelti: numero vicino, stesso numero di un
     altro codice, norma abrogata, norma simile non applicabile;
  3. fonte decisiva assente.
* **Validazione umana:** almeno 50 casi rivisti da un giurista. Le conclusioni
  scritte dal teacher non sono affidabili: 3 casi pratici su 6 erano sbagliati
  nel diritto (MEM, 29-30 ottobre). Si tengono solo i casi confermati.
* **Accordo del giudice per requisito**, sul campione validato. Sotto l'80% su
  un requisito, quel requisito non diventa una reward (fase 4, C3).
* **Esecuzione:**
  - r2l, g9r-r2, g4-r3 nelle tre condizioni;
  - i teacher (Qwen3.6-27B con e senza reasoning, Qwen3-30B) sugli stessi casi,
    caso per caso: quali casi il teacher risolve e lo studente no.
* **Diagnosi richiesta:** quota di errori di **scelta della fonte** (condizione
  2 peggio della 1) contro errori di **applicazione** (sbagliati anche nella 1).
* **Da scrivere:**
  - un generatore di casi;
  - il montaggio delle tre condizioni;
  - il giudice per requisito.

  Si riusano `legal_eval.py`, `compare_graded.py`, `citation_check.py`,
  `sbatch_teacher_eval.slurm`.
* **Costo:** circa 30-50 GPU-ore, più il tempo del giurista.

## Fase 2: E1-mini, memoria con il continued pretraining aumentato

**Ingresso:** fase 0.2 fatta. Può procedere in parallelo alla fase 1.

* **Backbone:** il vincitore di 0.2.
* **Metodo:** LoRA con **replay** del post-training in ogni batch (coppie
  open-book v06, premesse, esempi pratici, circa 20-30%) e 20% di testo
  italiano generale.
* **Dati:** il corpus degli articoli dei 9 codici: righe di programma, più le
  riscritture del teacher in parafrasi, domande e risposte, elementi. Circa
  40 milioni di token, 3-5 epoche. Le massime senza id solo se 0.1 le ha
  portate.
  - **Esclusi:** casi pratici e spiegazioni del teacher (3 su 6 sbagliati);
    le forme `domande` per gli articoli chiesti dalle valutazioni (già esclusi
    alla generazione).
* **Misura:**
  - libro chiuso, per tipo: giuste, sbagliate e sicure, articoli veri
    riconosciuti, numero ↔ rubrica, termini, pene, procedibilità rispetto
    alla base "querela";
  - devbig con e senza nota, pratico, trappole su 33, test di regressione 0.4.
* **Arresto:**
  - avanti solo se almeno **+15 punti** sul libro chiuso, contati **senza il
    riflesso** "non esiste" (articoli veri riconosciuti e numero ↔ rubrica,
    non le negazioni);
  - **nessuna perdita significativa** (McNemar, stretto e largo) a libro
    aperto e sulle trappole.

  Se non passa, si registra e ci si ferma: niente E1 a 2 miliardi di token.
* **Costo:** circa 20-50 GPU-ore.
* **Se passa:** prima del CPT a 2 miliardi di token serve il lavoro sui dati:
  - riscritture di tutti gli atti in vigore e degli atti UE in italiano
    (`law_rewrites.py` funziona già con qualsiasi raccolta);
  - un libro chiuso esteso a quegli atti.

  Il corpus attuale è circa 0,2 miliardi di token, non 2.

## Fase 3: E2-mini, prototipo del modulo n-gram

**Ingresso:** il codice del modulo scritto e testato; meglio dopo E1-mini.

* **Modulo, in Forge:**
  - tabelle hashed dei 2-gram e 3-gram;
  - lettura a uno strato basso, proiezione, gate inizializzato a zero;
  - ottimizzatore sparso;
  - tabella salvata come file separato e versionato.
* **Prova:** backbone **congelato** g4-r3 (il 4B, per costare poco), tabella da
  0,3-0,5 miliardi di parametri, stessi dati di E1-mini.
* **Controllo:** una LoRA con gli stessi parametri addestrabili e gli stessi
  token.
* **Misura:**
  - libro chiuso;
  - esami a libro aperto (con il backbone congelato non dovrebbero muoversi);
  - RAM, VRAM e token/s in inferenza.
* **Arresto:** se il modulo non batte il controllo di almeno 5 punti sul libro
  chiuso, le 1.800 GPU-ore di E2 non si spendono.
* **Costo:** 20-40 GPU-ore, più alcuni giorni di sviluppo.
* L'operatore nel motore (E4) si fa su LUMI solo se E2-mini passa.

## Fase 4: C2 oppure C3, secondo la diagnosi di C1

**C2, hard negatives:** se in C1 domina la scelta della fonte.
* **Generatore di distrattori**, da scrivere: numero vicino, stesso numero in
  un altro codice, abrogata, simile non applicabile.
* **SFT corto:** 2-4 mila righe, rifiuti a formula fissa sotto il 6% (regola
  di EXP §8). Poi un round di GRPO.
* **Misura:** C1, condizione 2, più regressione e trappole. Confronto con il
  backbone di partenza sullo stesso benchmark.
* **Costo:** circa 30-40 GPU-ore.

**C3, reward per requisito:** se gli errori restano anche con le sole fonti
giuste, e solo sui requisiti in cui il giudice regge (≥80%, fase 1).
* **Reward:**
  - una per requisito;
  - una di coerenza controfattuale (la conclusione cambia sul gemello), che
    richiede una modifica al calcolo delle reward in `grpo_train.py`;
  - prompt nuovi e difficili, presi dai casi C1 che il backbone sbaglia (fuori
    dal set di test C1).
* **Controllo:** un round con il voto unico attuale, stesso seme e stessi
  prompt.
* **Prima del round:** le reward vanno verificate su risposte vere in tutte e
  due le direzioni.
* **Costo:** circa 50-80 GPU-ore a round.

## Fase 5: scaling, solo se i cancelli sono passati

| Cosa | Cancello |
|---|---|
| E1 a 2B token (~120 GPU-ore), poi 10B | E1-mini passato, dati di tutti gli atti pronti, libro chiuso esteso |
| E2 in tre taglie (3 × 600 GPU-ore) | E2-mini sopra il controllo |
| Altri round C2/C3 | miglioramento su C1 senza regressioni |
| E5, lookup appreso | E1 o E2 utili, C1 stabile |

La Fast Lane su Leonardo (EHPC-AIF-2026FL01-1400, 30.000 GPU-ore fino al 2
febbraio 2027) è stata concessa il 2 novembre 2026: lo scaling va lì, e anche
le fasi 0-4 possono girarci. Divisione del lavoro con JUPITER, ritmo e
traguardi sono in `docs/cineca/fastlane-plan.md`.

## Regole trasversali

* **Versioni:**
  - ogni training ha un nome e registra il suo punto di partenza;
  - mai due training che partono dallo stesso checkpoint senza un nome
    diverso e un confronto previsto;
  - un training lungo sopra un modello già passato dal GRPO è seguito dagli
    esami a libro aperto prima di essere usato.
* **Contaminazione:**
  - articoli e casi di C1, libro chiuso, held-out e LexIA mai nei dati di
    training né nei prompt dei teacher;
  - l'esame held-out si chiede una volta per candidato, e se ne riportano
    solo i totali.
* **Dati personali:** come 0.1, fino all'addendum. Prima di rilasciare un
  modello addestrato su dati derivati dalle sentenze: test di estrazione e
  canary (MEM, "Privacy").
* **Letture:** prima di ogni decisione si leggono almeno cinque risposte per
  direzione; quello che una lettura smentisce va nel registro accanto alla
  conclusione che sostituisce.

## Ordine e costi in breve

| Fase | GPU-ore | Dipende da |
|---|---|---|
| 0.2 confronto backbone | 10-20 | — |
| 0.3 indice esteso | < 10 | — |
| 0.4 regressione | < 5 | — |
| 1 C1 | 30-50 + giurista | 0.2 |
| 2 E1-mini | 20-50 | 0.2 |
| 3 E2-mini | 20-40 + sviluppo | codice del modulo |
| 4 C2 o C3 | 30-80 | 1 |
| **Totale fino ai cancelli di scaling** | **circa 150-250** | |
