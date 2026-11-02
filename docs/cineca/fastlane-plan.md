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
| Calendario | 92 giorni; **nessuna proroga** ("extensions … are not granted") |
| Inizio | l'uso deve cominciare entro il primo mese; CINECA può applicare un profilo di consumo prestabilito |
| Dichiarato nella proposta | fino a 16 GPU contemporanee, 4 TB di spazio, 2 TB di trasferimenti, nessun membro del team oltre al PI |
| Dati personali | "we will complete the required agreement before such processing takes place": **nessuna sentenza né dato derivato su questo progetto prima del DPA con CINECA** |
| Report finale | obbligatorio, sul modello del sito EuroHPC JU |

## Cosa abbiamo promesso

Il report finale e la domanda Large Scale che segue sono giudicati su questi
punti. Ogni ora spesa deve lasciare un numero in una di queste righe.

| # | Impegno (dalla proposta) | Dove si misura |
|---|---|---|
| P1 | costo misurato di un ciclo completo di verticalizzazione: teacher, CPT, distillazione, dati sintetici, instruction tuning, valutazione, GGUF | WP6, più le ore per fase di tutti i WP |
| P2 | confronto sistematico di taglie di teacher e di student e di configurazioni di distillazione | WP3 |
| P3 | generazione e filtraggio di dati sintetici | WP2 |
| P4 | fino a 16 A100 in parallelo: quanto il flusso regge passando da un esperimento a una campagna su più nodi | il registro di coda (`queue_stats.py`) e l'occupazione |
| P5 | ogni modello valutato su dati esclusi dal training e confrontato con la base; regressioni bloccate | WP1 |
| P6 | configurazioni riusabili per altri domini | WP6, secondo dominio |
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
     teacher è fisso e gli student variano, cioè esattamente WP3): gli anelli
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
4. **Il venv si ricrea, non si copia:** cambia il percorso e gli script del
   venv lo hanno scritto dentro. Si usa
   `pip install -r eullm_venv-known-good-2026-10-01.txt`, mai versioni a memoria.
5. **Gli script:** `sbatch_queue_stats.slurm` e `sbatch_transfer_curve.slurm`
   hanno il percorso della 1147 come valore predefinito, quindi si passano
   `EULLM_BASE` e `CURVE_RUN_DIR`. Il resto legge `EULLM_ACCOUNT`.
6. **Il registro:** `sbatch_queue_stats.slurm` in catena da subito. `sacct`
   perde la storia dopo qualche settimana, e il report deve citare intervalli
   datati.

## Il lavoro, in pacchetti

| WP | Cosa | GPU-ore | Cancello | Impegni |
|---|---|---:|---|---|
| WP0 | messa in opera, sondaggi, registro | 50 | — | P4 |
| WP1 | **valutazione**: test di regressione (roadmap 0.4), confronto r2l / g9r-r2 sugli stessi esami (0.2), indice esteso (0.3), benchmark C1, valutazione di ogni checkpoint, giudice | 2.500 | — | P5 |
| WP2 | **output del teacher**: (a) riscritture di tutti gli atti in vigore e degli atti UE in italiano (i dati di E1 oltre il corpus attuale di ~0,2 miliardi di token); (b) cache offline dei logit del 30B-A3B su norms-v2, e sulle sentenze dopo il DPA; (c) generazione dei casi C1 e dei distrattori C2 | 6.000 | (b) sentenze: DPA | P3, P1 |
| WP3 | **matrice degli student**, il cuore della proposta: Qwen3.5-4B e 9B e Ministral 8B, con distillazione sui logit, KD a livello di sequenza e solo SFT a parità di GPU-ore; rank 32 contro 128; LoRA contro fine-tuning completo sul 4B; temperatura della KL | 8.000 | — | P2, P1 |
| WP4 | **memoria**: E1-mini, poi E1 a 2 e a 10 miliardi di token; E2-mini, poi E2 in tre taglie; E3, la riforma simulata | 5.000 | E1-mini +15 a libro chiuso; E2-mini +5 sul controllo | P2 |
| WP5 | **ragionamento**: C2 oppure C3, secondo la diagnosi di C1 | 2.000 | C1 costruito e validato dal giurista | P5 |
| WP6 | **ciclo completo cronometrato**: legal-it in due taglie, dal corpus al GGUF alla valutazione, senza intervento; poi un secondo dominio con la stessa configurazione | 3.000 | secondo dominio: scelta dell'utente | P1, P6 |
| — | riserva: riprese, rifacimenti, errori | 3.450 | — | |
| | **Totale** | **30.000** | | |

**WP0-WP3 e la parte legal-it di WP6 non hanno cancelli** (~19.500 GPU-ore):
sono il lavoro che assorbe la coda dalla prima settimana, mentre WP4 e WP5
aspettano le loro letture.

I cancelli restano quelli della roadmap. La Fast Lane non è un motivo per
scalare un esperimento che non li ha passati: le ore di un cancello fallito
passano a WP3 e WP6, che le usano comunque per gli impegni P1 e P2.

## Calendario

| Settimane | Lavoro |
|---|---|
| 1 (2-8 nov) | WP0; copia interna dei dati; WP1: 0.2, 0.3 e 0.4 con job da una GPU; WP2a in catena appena l'indice degli atti in vigore è pronto (`prepare_legislation.py --all-acts`, lavoro CPU sulla seriale) |
| 2-4 (9-29 nov) | WP2a e WP2b; prime braccia di WP3 sulla cache dei logit; E1-mini (WP4); costruzione di C1, più il tempo del giurista |
| 5-8 (30 nov-27 dic) | WP3 a regime; E1 a 2 miliardi di token se E1-mini è passato; E2-mini; C1 su backbone e teacher, poi la diagnosi; primo round C2 o C3 |
| 9-11 (28 dic-17 gen) | si scalano i vincitori (E1 a 10 miliardi, E2 in tre taglie, altri round C); WP6 legal-it cronometrato; secondo dominio |
| 12-13 (18 gen-1 feb) | valutazioni finali, GGUF, misure per il report. **Dopo il 25 gennaio non parte niente che non finisca entro il 1° febbraio.** Copia dei risultati fuori da Leonardo |

## Leonardo e JUPITER

JUPITER (1434) resta attivo fino al 31 dicembre. La divisione:

* **JUPITER:** quello che la roadmap gli dà, sui dati ammessi lì, e i suoi impegni GH200
  (`../jupiter/allocation-plan.md`).
* **Leonardo:** la scala, la matrice degli student e, dopo il DPA, tutto
  quello che deriva dalle sentenze.
* **Mai lo stesso esperimento su tutti e due**, salvo come confronto di
  piattaforma voluto. Ogni esperimento ha una sola casa, scritta qui o nel
  dossier di JUPITER.

## Linee da non oltrepassare

* **Sentenze e dati derivati solo dopo il DPA con CINECA**, come scritto nella
  proposta. Fino ad allora: norme, corpus memoria degli articoli, r2l/r2q,
  libro chiuso, devbig, pratico, trappole. Gli impegni sulla
  pseudonimizzazione sono gli stessi di JUPITER (`../jupiter/allocation-plan.md`,
  "Lines not to cross").
* **Contaminazione:** come nella roadmap ("Regole trasversali").
* **Solo il PI sull'account**, finché CINECA non aggiunge un membro.
* **Nessuna ora senza un risultato** legato a una riga di "Cosa abbiamo promesso".
* **Il 2 febbraio è fisso:** il piano di uscita dei dati si scrive a gennaio,
  non l'ultimo giorno come per la 1147.

## Domande aperte per CINECA (primo messaggio)

1. Collegare o lasciare leggibile `$WORK` di AIFAC_P02_1147 fino alla copia
   interna; sorte di `$SCRATCH`.
2. Profilo di consumo prestabilito, QOS del progetto, limite di GPU
   contemporanee, priorità dei job.
3. Procedura per il DPA (dati giudiziari pseudonimizzati).
4. Procedura per aggiungere un membro del team.
5. Consumo energetico per job, per il report.

## Report finale

Si raccoglie dal primo giorno, non alla fine:
* GPU-ore per fase, da `sacct`, con un prefisso di nome per WP;
* il registro di coda con le attese datate;
* le tabelle dei risultati con la base di confronto;
* il costo di un ciclo completo (WP6).

Il modello va scaricato dal sito EuroHPC JU prima di gennaio.
