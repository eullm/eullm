# Piano Fast Lane — EHPC-AIF-2026FL01-1400 (Leonardo Booster)

> **30.000 GPU-ore su Leonardo Booster (A100 64 GB), dal 2 novembre 2026 al
> 2 febbraio 2027.** EuroHPC AI Factory Fast Lane, assegnata il 2 novembre
> 2026 con tutte le ore chieste. Proposta: *Scaling EULLM Forge for Systematic
> Training of European Domain-Specific Language Models*. Valutazione tecnica:
> "Accepted", con l'argomento che il lavoro è già stato provato su Leonardo con
> la Playground EHPC-AIF-2026PG01-1147.
>
> Altre assegnazioni attive:
> * JUPITER Booster EHPC-AIF-2026PG01-1434, fino al 31 dicembre 2026
>   ([`../jupiter/allocation-plan.md`](../jupiter/allocation-plan.md));
> * LUMI-G EHPC-DEV-2026D09-278, solo inferenza, fino al 12 marzo 2027.
>
> Il lavoro da fare è in [`../research/legal-it-roadmap.md`](../research/legal-it-roadmap.md)
> e [`../research/legal-memory-plan.md`](../research/legal-memory-plan.md);
> cosa è già stato provato è in [`../research/legal-it-audit-2026-11.md`](../research/legal-it-audit-2026-11.md).

## Le condizioni dell'assegnazione

| | |
|---|---|
| Ore | 30.000 GPU-ore = 7.500 ore nodo = 240.000 ore locali di `saldo` (8 core per GPU) |
| Calendario | 92 giorni; **nessuna proroga** ("extensions … are not granted"). CINECA (2 novembre): il progetto parte alla data ufficiale o quando la registrazione è completa, e **finisce comunque il 2 febbraio** |
| Ritmo | la *budget linearization policy* di CINECA: quota mensile = totale / 3 = **80.000 ore locali = 10.000 GPU-ore al mese**. Sotto la quota la priorità è più alta, sopra scende; quello che avanza a fine mese resta, ma con priorità più bassa |
| Spazio | `$WORK` parte da **1 TB**: per arrivare ai 4 TB della proposta va chiesto a superc@cineca.it a progetto avviato |
| Collaboratori | li aggiunge il PI da UserDB ("My Projects" → progetto → "edit" → "collaborators"), dopo che si sono registrati loro stessi |
| Dichiarato nella proposta | fino a 16 GPU contemporanee, 4 TB di spazio, 2 TB di trasferimenti, nessun membro del team oltre al PI |
| Dati personali | Il modulo: "they will need to make a data processing agreement with the production site". La nostra risposta: "a data processing agreement … **may be required** before personal data are uploaded or processed on the system, and we will complete the required agreement before such processing takes place". **L'impegno è condizionato**: l'accordo si firma se il sito lo chiede. CINECA ci ha attivati sapendo dei dati personali e senza chiederlo (JSC invece ha atteso l'accordo). Le sentenze sono su Leonardo dalla 1147. Decisione del PI (3 novembre 2026): si lavora; se CINECA chiederà un accordo, lo si fa allora |
| Report finale | obbligatorio, sul modello del sito EuroHPC JU |

## Cosa abbiamo promesso

Il report finale e la domanda Large Scale che segue sono giudicati su questi
punti. Ogni ora spesa deve lasciare un numero in una di queste righe.

| # | Impegno (dalla proposta) | Dove si misura |
|---|---|---|
| P1 | costo misurato di un ciclo completo di verticalizzazione: teacher, CPT, distillazione, dati sintetici, instruction tuning, valutazione, GGUF | C5, più le ore per fase di tutte le linee |
| P2 | confronto sistematico di taglie di teacher e di student e di configurazioni di distillazione | C3; A per il MoE contro il denso |
| P3 | generazione e filtraggio di dati sintetici | C2 |
| P4 | fino a 16 A100 in parallelo: quanto il flusso regge passando da un esperimento a una campagna su più nodi | B (scalabilità su 1-4 nodi), più il registro di coda (`queue_stats.py`) e l'occupazione |
| P5 | ogni modello valutato su dati esclusi dal training e confrontato con la base; regressioni bloccate | C1 |
| P6 | configurazioni riusabili per altri domini | C5, secondo dominio |
| P7 | la base numerica per la Large Scale: ore GPU osservate, non stimate | tutto il registro |

## Il numero che governa tutto

30.000 GPU-ore in 92 giorni sono **326 GPU-ore al giorno**. Le 16 GPU
dichiarate, accese giorno e notte, fanno 384: serve tenerle occupate all'85%
per tre mesi.

Sulla 1147 non ci siamo mai andati vicino:
* le giornate migliori (fine settembre) erano 40-50 ore nodo, cioè 160-200 GPU-ore;
* negli ultimi undici giorni la media è stata circa 23 ore nodo, cioè 90 GPU-ore.

**Serve quindi 1,6 volte la nostra giornata migliore, tutti i giorni.** Il
rischio della Fast Lane non è restare senza idee: è restare indietro con la coda.

Le leve, misurate sulla 1147 (`allocation-recovery.md`):

1. **Il numero di catene concorrenti** è l'unica leva lineare. Con anelli da
   3 GPU servono almeno 6-8 catene vive in ogni momento, più job da una GPU
   (valutazioni, giudice, SFT corti) che riempiono i buchi.
2. **Il rendimento di un anello.** Su un anello da 2 ore se ne vanno ~14 minuti
   all'avvio: 3 per gli import, 11 per caricare il teacher da 61 GB. È il 20%
   delle ore, cioè **~6.000 GPU-ore** dell'assegnazione se non cambia niente.
   Si recuperano in due modi:
   * la cache offline dei logit del teacher (ADR-001: conviene quando il
     teacher è fisso e gli student variano, cioè esattamente C3): gli anelli
     degli student non caricano più il teacher;
   * anelli più lunghi, se il sondaggio del primo giorno dice che il nuovo
     progetto li fa partire.
3. **Il limite di GPU.** Se CINECA applica davvero un tetto di 16 GPU
   (`GrpTRES`), l'85% di occupazione è il massimo teorico: lo si chiede e lo si
   legge il primo giorno.
4. **Lavoro sempre pronto.** In lista ci devono essere sempre almeno due
   settimane di job pronti da accodare (la tabella dei WP qui sotto). Una
   catena che finisce va sostituita nello stesso messaggio (regola 8 di
   `forge/CLAUDE.md`).

### Traguardi cumulativi

| Data | GPU-ore spese | Lineare |
|---|---:|---:|
| 15 novembre | 3.000 | 4.200 |
| 1 dicembre | 9.000 | 9.400 |
| 15 dicembre | 14.500 | 14.000 |
| 1 gennaio | 20.000 | 19.500 |
| 15 gennaio | 25.000 | 24.100 |
| 1 febbraio | 30.000 | 29.700 |

La prima quindicina sta sotto la linea perché c'è la messa in opera. Ogni
lunedì si confronta `budget.sh` con questa tabella. **Più del 10% sotto il
traguardo vuol dire due catene in più dalla lista, quella settimana stessa**,
non "si recupera dopo": il 2 febbraio non si sposta.

**Perché la linea conta due volte.** Con la *linearization policy* le ore
non spese in un mese passano al mese dopo con priorità più bassa: chi resta
indietro a novembre aspetta di più in coda proprio a dicembre, quando deve
recuperare. Superare la quota del mese abbassa la priorità allo stesso modo.
Il traguardo di ogni mese è quindi la sua quota, 10.000 GPU-ore, non di più
e non di meno: è quello che `saldo -b` mostra come quota mensile.

## Il primo giorno, prima di qualsiasi training

1. **Il progetto.** Leggere `saldo -b` per il nome del conto, poi in `~/.bashrc`:
   ```bash
   export EULLM_ACCOUNT=<nuovo conto>
   export EULLM_BUDGET_HOURS=240000
   export EULLM_BUDGET_START=2026-11-02
   export EULLM_BUDGET_END=2027-02-02
   ```
   Poi leggere i limiti veri:
   ```bash
   sacctmgr -P show assoc user=$USER format=account,partition,qos,grptres,maxtres,maxjobs,maxsubmit
   sacctmgr -P show qos format=name,priority,maxwall,grptres,maxtrespu,maxjobspu
   ```
2. **Il sondaggio di collocazione**: un comando solo, dal nodo di login.
   ```bash
   module load python/3.11.7
   bash forge/scripts/leonardo/fastlane_probe.sh
   ```
   * Scrive subito `limits.txt`: QOS del progetto, priorità, wall time
     massimo, limiti di GPU e di job.
   * Mette in coda un job di prova per ogni forma:
     - 1 e 3 GPU;
     - nodo intero;
     - 2, 4 e 8 nodi;
     - da 2 a 24 ore;
     - con ogni QOS del progetto.

     I job escono appena partono e costano secondi.
   * Mette in coda anche due job da 16 GPU che restano su 15 minuti ciascuno,
     per vedere se 16 + 16 girano insieme (circa 8 GPU-ore).
   * Il report (`report.md`) lo scrivono job sulla partizione seriale a +1, +6
     e +24 ore; l'ultimo cancella le prove ancora in coda. In tutto costa circa
     10 GPU-ore.

   | Se il report dice | Allora |
   |---|---|
   | nodi interi da 8-24 ore partono in poche ore | catene di nodi interi lunghe: rendimento ~99% invece di ~80%, bastano 3-4 nodi sempre occupati |
   | partono solo 3 GPU per 2 ore, come sulla 1147 | 6-8 catene concorrenti, cache dei logit, e **subito** una richiesta a CINECA di una reservation o di una QOS adatta, citando le 16 GPU della proposta |
   | 16 + 16 girano insieme | il tetto sale a 768 GPU-ore al giorno: basta un'occupazione del 42% invece dell'85% |
   | un motivo di limite (`QOS…Limit`, `Assoc…`) | quella forma non è permessa: non si aspetta, si chiede a CINECA |

   **Il primo `limits.txt` (3 novembre 2026, 11:04):**
   * il conto `aifac_f02_1400` non ha limiti propri: nessun tetto di GPU, di
     job o di wall time. Le 16 GPU sono solo la nostra dichiarazione;
   * le QOS sono le stesse della 1147, senza priorità per la AI Factory:
     - `normal`: priorità 40; la partizione permette fino a 24 ore e 64 nodi per job;
     - `boost_qos_lprod`: priorità 40, fino a **4 giorni**, al massimo 8 nodi
       e 32 GPU per utente;
     - `boost_qos_dbg`: priorità 80, 30 minuti, 2 job;
     - `boost_qos_bprod`: solo da 65 nodi in su (le tre prove sono state
       rifiutate con `QOSMinCpuNotSatisfied`);
   * la macchina: 2.872 nodi pieni, 327 occupati in parte, 30 tenuti per job
     pianificati, **nessuno libero**. È lo stesso quadro di settembre.

   **Le prime prove (3 novembre 2026, inviate alle 11:04, lette alle 11:47):**
   * **partite:** 1 GPU 2 h; 3 GPU a 2, 4, 8 e **24 h**, anche con
     `boost_qos_lprod`; nodo intero 2 h; **4 nodi 2 h**;
   * **16 + 16 confermato**: par-a e par-b (4 nodi ciascuno) partiti tutti e
     due alle 11:05:38, 98 secondi dopo l'invio, e finiti insieme alle
     11:20:38, quindi 32 GPU contemporanee;
   * **in coda (Priority):** nodo intero 8 e 24 h, 2 nodi 8 h, 4 nodi 8 e
     24 h, 8 nodi 2 h, e le due forme `lprod` a nodo intero.

   Rispetto a settembre, i job da 3 GPU partono a qualunque durata fino a 24
   ore. Le regole di `forge/CLAUDE.md` (2 ore, mai il nodo intero) si
   cambiano dopo il report finale delle prove, non su una lettura delle 11.

   La lunghezza degli anelli si decide su questo, non sull'abitudine. Finché
   il sondaggio non ha risposto valgono le regole di `forge/CLAUDE.md` e
   `test_leonardo_job_shape.py`.
3. **I dati.** La vecchia area `/leonardo_work/AIFAC_P02_1147` è ancora
   leggibile: si copia dentro Leonardo nella nuova area, senza passare
   dall'esterno, prima che l'accesso si chiuda. Va copiato solo quello che
   serve: `eullm_runs` senza i checkpoint intermedi, i dataset, le valutazioni,
   le norme. I modelli base vanno sulla nuova scratch veloce. Si controlla anche
   se `grpo-merged` è ancora sulla scratch; se non c'è, l'unica copia è sul VPS
   temporaneo.
   La copia la fa un job sulla partizione seriale, che prima misura
   (`COPY_DRY=1`) e alla fine verifica se stesso ("TUTTO COPIATO"):
   `forge/scripts/leonardo/sbatch_copy_project.slurm`. Sul login rsync lunghi
   sono stati uccisi dal limite di CPU (ottobre). L'area della 1147 resta
   comunque per sei mesi (UserDB, "ExpirationDelay: 6").
4. **Il venv si ricrea, non si copia:** cambia il percorso e gli script del
   venv lo hanno scritto dentro. Si usa
   `pip install -r eullm_venv-known-good-2026-10-01.txt`, mai versioni a memoria.
5. **Gli script:** `sbatch_queue_stats.slurm` e `sbatch_transfer_curve.slurm`
   hanno il percorso della 1147 come valore predefinito, quindi si passano
   `EULLM_BASE` e `CURVE_RUN_DIR`. Il resto legge `EULLM_ACCOUNT`.
6. **Il registro:** `sbatch_queue_stats.slurm` in catena da subito. `sacct`
   perde la storia dopo qualche settimana, e il report deve citare intervalli
   datati.
7. **Lo spazio:** `$WORK` nasce da 1 TB. Lo stesso giorno si scrive a
   superc@cineca.it per portarlo ai 4 TB assegnati, prima di copiare i dati
   della 1147. Sulla 1147 un solo TB era arrivato al 96% (10 ottobre).

## Le priorità (decise il 2 novembre 2026)

1. **Un MoE con il modulo n-gram "mini" sul legal-it** (linea A).
2. **La validazione di Forge su più nodi** (linea B).
3. Tutto il resto, cioè gli impegni della proposta che riempiono la coda (linea C).

L'ordine è di importanza, non di ore. La linea A nella fase mini costa poco
(circa 1.000 GPU-ore) ma decide dove vanno le 7.000 della sua fase di scala.
La linea C è quella che tiene le GPU occupate mentre A e B leggono i loro
risultati.

### Linea A — MoE con memoria n-gram sul legal-it

**Perché un MoE.** Il modulo n-gram (`../research/legal-memory-plan.md`,
"Architecture") è una memoria condizionale: si consulta per hash, senza
calcolo. Il MoE fa la stessa cosa con il calcolo: tanti parametri, pochi
attivi per token. I lavori da cui viene l'idea (DeepSeek Engram, gli embedding
n-gram di Qwen3.8-Flash-Next) mettono la memoria proprio sopra un MoE.

In locale un 30B-A3B in Q4 occupa ~18 GB e gira con 3B parametri attivi:
sta su una GPU da 24 GB, oppure su CPU e RAM a velocità usabile. La tabella
sta in RAM.

**Il backbone.** Candidato: **Qwen3-30B-A3B-Instruct-2507** (Apache 2.0). Su
Leonardo lo conosciamo già: giudice e teacher, e la fase 1 di settembre ci ha
insegnato come addestrarlo sopra un modello congelato
(`continued_pt_qwen3_30b_a3b_leonardo.yaml`):
* pesi quantizzati a 8 bit e replicati su ogni GPU;
* parallelismo sui dati;
* **mai ZeRO-3**, che su un MoE muove 625M parametri per strato e ha dato tre OOM.

Un MoE più piccolo o più recente si valuta in A0, con la licenza verificata
sull'Hub (regole in `forge/CLAUDE.md`).

| Passo | Cosa | GPU-ore | Esito |
|---|---|---:|---|
| A0 | il backbone sugli esami attuali: libro chiuso 716, devbig con e senza nota, pratico 145, trappole 33, regressione | 50 | la base di confronto |
| A1 | **legal-it-moe**: LoRA open-book sul MoE con la ricetta che ha dato 80,9 → 84,6 sull'8B (coppie v06, solo norme) | 250 | un MoE che legge e cita come i nostri, prima di dargli memoria |
| A2 | **E2-mini sul MoE**: A1 congelato più una tabella n-gram di 0,3-0,5 miliardi di parametri (2- e 3-gram hashed, lettura a uno strato basso, gate inizializzato a zero, ottimizzatore sparso). Dati: il corpus memoria degli articoli (righe di programma più riscritture, ~0,2 miliardi di token, 3 epoche). **Controllo**: una LoRA con gli stessi parametri addestrabili e gli stessi token. **Riferimento**: r2l denso sullo stesso libro chiuso | 700 | **cancello**: il modulo batte il controllo di ≥ 5 punti sul libro chiuso, contati senza il riflesso "non esiste", e nessuna perdita a libro aperto (McNemar) |
| A3 | se A2 passa: tabelle da 1 e 3 miliardi; strato e ordini degli n-gram; i dati di tutti gli atti in vigore e degli atti UE (C2); E3, la riforma simulata (si riaddestra solo la memoria); i principi di diritto | 7.000 | la tabella vincente, con la curva taglia/beneficio |
| A4 | il modello in locale: tabella come tensori GGUF e l'operatore di lookup nel motore (E4 del piano memoria, sul lato LUMI) | — | misurato sul portatile, non solo sui benchmark |

**Il MoE con transformers 5.22.1 (prova `venv_smoke.py`, job 63412010, 3
novembre 2026).** I 128 esperti di ogni strato sono due tensori fusi
(`gate_up_proj`, `down_proj` in `Qwen3MoeExperts`), non più strati lineari,
e bitsandbytes quantizza solo gli strati lineari. Il "30B a 8 bit" pesa
quindi 55,2 GiB in bf16 più 0,8 GiB in int8 (picco 56,8 GiB): riempie da
solo un'A100 da 64 GB. La ricetta di settembre (8 bit replicati su ogni GPU)
non vale più. In cambio, gli esperti passano per `grouped_mm`, una sola
moltiplicazione per strato su A100 con torch ≥ 2.10, invece del ciclo per
esperto di settembre.

Per A1-A2 il MoE in bf16 si divide fra le GPU del nodo (FSDP, ~15 GB a GPU
su 4): resta spazio per la tabella e le attivazioni, e lo stesso montaggio
si estende a più nodi, quindi è anche il carico della linea B. Nella stessa
prova la GPU fa 292 TFLOP/s in bf16, e Qwen3.5-4B fa il primo forward in
8,65 s (compilazione Triton) e il secondo in 0,35 s.

**Se A2 non passa** si scrive il risultato, si ferma la linea, e le 7.000 ore
di A3 vanno a C3 e C5 (o a E1, il pretraining continuato del denso con
replay, se il libro chiuso dice che manca la conoscenza e non la
struttura).

**Codice da scrivere prima che il conto sia attivo**, così le GPU non aspettano lo sviluppo:
* il modulo in Forge: tabella, hash, gate, iniezione nello strato, ottimizzatore
  sparso, salvataggio come file separato e versionato;
* i test su CPU con un modello minuscolo:
  - a gate zero il modello è identico all'originale;
  - i gradienti arrivano solo alla tabella;
  - il file della memoria si ricarica identico;
* il controllo LoRA con parametri pareggiati.

### Linea B — Forge su più nodi

Oggi ogni training di Forge sta in un nodo. Più nodi servono per due motivi:
* la proposta promette fino a 16 A100 insieme (P4);
* la domanda Large Scale chiederà la scalabilità misurata, non dichiarata.

| Passo | Cosa | GPU-ore | Esito |
|---|---|---:|---|
| B0 | lanciatore multinodo: `srun` più `torchrun` con rendezvous sul primo nodo, variabili NCCL per l'InfiniBand di Leonardo; benchmark di all-reduce in PyTorch su 1, 2, 4 e 8 nodi | 50 | banda reale fra nodi |
| B1 | **scalabilità sul carico vero**: il training di A2 su 1, 2 e 4 nodi a batch globale fisso; token al secondo per GPU, efficienza, tempo di checkpoint. La domanda tecnica: la tabella n-gram con gradienti sparsi in DDP, oppure divisa fra i nodi | 400 | la curva di scalabilità per il report e per la Large Scale |
| B2 | **robustezza**: ripresa dopo un anello interrotto su più nodi, dati compresi (la regola 5 di `allocation-recovery.md`), checkpoint condivisi | 100 | i training lunghi di A3 e C3 possono girare multinodo |
| B3 | gli stessi passi per un training con pesi addestrati (SFT completo del 4B, C3), dove la comunicazione pesa davvero | 450 | quando conviene il multinodo e quando no |

**B dipende dal test del primo giorno.** I job multinodo chiedono nodi interi.
Se il test dice che i nodi interi non partono, B1-B3 si fanno solo con una
prenotazione dei nodi, che va chiesta a CINECA per finestre di poche ore
indicando che è la misura promessa nella proposta. Lo script multinodo va
anche aggiunto, con la sua ragione, all'elenco `WHOLE_NODE` di
`test_leonardo_job_shape.py`.

### Linea C — gli impegni che riempiono la coda

| | Cosa | GPU-ore | Cancello | Impegni |
|---|---|---:|---|---|
| C0 | messa in opera, test del primo giorno, registro | 50 | — | P4 |
| C1 | **valutazione**: test di regressione (roadmap 0.4), confronto r2l / g9r-r2 (0.2), indice esteso (0.3), benchmark C1, ogni checkpoint di A, B e C | 2.500 | — | P5 |
| C2 | **output del teacher**: riscritture di tutti gli atti in vigore e degli atti UE (i dati di A3); cache dei logit; casi C1 e distrattori; principi delle sentenze | 5.000 | — | P3 |
| C3 | **matrice degli student**: 4B, 9B e 8B × KD sui logit, KD di sequenza, solo SFT, a parità di GPU-ore; rank; LoRA contro completo | 5.000 | — | P2 |
| C4 | **ragionamento**: C2 oppure C3 della roadmap, secondo la diagnosi del benchmark C1 | 2.000 | C1 validato dal giurista | P5 |
| C5 | **ciclo completo cronometrato**: legal-it dal corpus al GGUF, senza intervento; poi un secondo dominio | 3.000 | secondo dominio: scelta dell'utente | P1, P6 |

### Il budget

| | GPU-ore |
|---|---:|
| A0-A2 (MoE n-gram mini) | 1.000 |
| A3 (scala, solo se A2 passa) | 7.000 |
| B (multinodo) | 1.000 |
| C0-C5 | 17.550 |
| riserva | 3.450 |
| **Totale** | **30.000** |

Senza cancelli ci sono ~17.500 GPU-ore (A0-A2, B, C0-C3, C5): bastano a
riempire la coda fino a dicembre inoltrato, quando A2 e C1 hanno già dato
il loro esito.

## Calendario

| Settimane | A — MoE n-gram | B — multinodo | C — coda |
|---|---|---|---|
| 0 (fino all'attivazione) | codice del modulo e test su CPU; scelta del backbone | lanciatore e benchmark di all-reduce scritti | — |
| 1 | A0 | B0 se i nodi interi partono | C0, test del primo giorno, copia dei dati; C1 (0.2-0.4); C2 in catena |
| 2-3 | A1, poi la prova del modulo su un nodo | B1 sul carico di A2 | C2, C3 |
| 4-5 (fino al 6 dic) | **A2 e lettura del cancello** | B2 | C2, C3; costruzione di C1 con il giurista |
| 6-9 (fino al 3 gen) | A3 se il cancello è passato, multinodo | B3 | C3, C4, C5 legal-it |
| 10-11 (fino al 17 gen) | A3: la taglia vincente sui dati completi | — | C5, secondo dominio |
| 12-13 (18 gen-1 feb) | GGUF, A4 su LUMI, misure per il report | numeri per la Large Scale | valutazioni finali; **dopo il 25 gennaio non parte niente che non finisca entro il 1° febbraio**; copia dei risultati fuori da Leonardo |

## Leonardo e JUPITER

JUPITER (1434) resta attivo fino al 31 dicembre. La divisione:

* **JUPITER:** quello che la roadmap gli dà, sui dati ammessi lì, e i suoi impegni GH200
  (`../jupiter/allocation-plan.md`).
* **Leonardo:** la scala, la matrice degli student e tutto quello che
  deriva dalle sentenze, che su JUPITER resta fuori fino all'addendum di JSC.
* **Mai lo stesso esperimento su tutti e due**, salvo come confronto di
  piattaforma voluto. Ogni esperimento ha una sola casa, scritta qui o nel
  dossier di JUPITER.

## Linee da non oltrepassare

* **Sentenze:** solo pseudonimizzate. Gli impegni della proposta sono gli
  stessi di JUPITER (`../jupiter/allocation-plan.md`, "Lines not to cross").
  Un accordo con CINECA si firma se CINECA lo chiede; non lo si chiede noi
  (decisione del PI, 3 novembre 2026).
* **Contaminazione:** come nella roadmap ("Regole trasversali").
* **Solo il PI sull'account**, finché CINECA non aggiunge un membro.
* **Nessuna ora senza un risultato** legato a una riga di "Cosa abbiamo promesso".
* **Il 2 febbraio è fisso:** il piano di uscita dei dati si scrive a gennaio,
  non l'ultimo giorno come per la 1147.

## Domande per CINECA

La prima mail è partita il 2 novembre 2026 (domande 1-2), la seconda il 3
novembre (domande 5 e 7).

1. Collegare o lasciare leggibile `$WORK` di AIFAC_P02_1147 fino alla copia
   interna; sorte di `$SCRATCH`. *Aperta.*
2. Profilo di consumo, QOS del progetto, limite di GPU contemporanee,
   priorità dei job. *In parte risposta*: la quota mensile e la priorità
   vengono dalla linearization policy. QOS e limiti si leggono con
   `fastlane_probe.sh`.
3. Accordo sul trattamento dei dati: *non chiesto*, per decisione del PI; se ne parla se lo chiede CINECA.
4. Procedura per aggiungere un membro del team. *Risposta*: da UserDB, la fa il PI.
5. Consumo energetico per job, per il report. *Inviata il 3 novembre.*
6. Data di fine allineata all'inizio effettivo. *Risposta: no*, finisce il 2
   febbraio comunque.
7. Aumento di `$WORK` a 4 TB (su EFP la richiesta risulta: storageRequest 4096 GB). *Inviata il 3 novembre.*

## Report finale

Si raccoglie dal primo giorno, non alla fine:
* GPU-ore per fase, da `sacct`, con un prefisso di nome per WP;
* il registro di coda con le attese datate;
* le tabelle dei risultati con la base di confronto;
* il costo di un ciclo completo (C5);
* la curva di scalabilità su più nodi (B1).

Il modello va scaricato dal sito EuroHPC JU prima di gennaio.
