# Fast Lane 1400: diario dell'infrastruttura

Quello che Leonardo ci ha dato e non dato, giorno per giorno, per la parte
del report finale sull'infrastruttura (`fastlane-plan.md`, "Report finale").
Solo fatti misurati, con la fonte di ciascuno. Nessun utente o progetto
terzo viene nominato: si descrive l'effetto sulla macchina, non chi l'ha
causato.

Le serie automatiche da cui si ricavano i numeri:
* **attese in coda**: `sbatch_queue_stats.slurm`, ogni 3 ore
  (`$WORK/eullm_runs/queue-stats/`), e le prove di forma
  (`fastlane_probe.sh`, rapporti in `$WORK/eullm_runs/fastlane-probe/<data>/report.md`);
* **carico dei nodi di login**: una riga a ogni nostro accesso e a ogni
  `carico` (`login.sh`, `$WORK/eullm_runs/login-load.tsv`);
* **ore spese e job morti**: `sacct`.

## 3 novembre 2026: primo giorno

Prova di forma alle 11:04 (`fastlane-probe/`, prima corsa):
* job da 3 GPU di 2-24 ore partiti in 13-20 s;
* un nodo intero per 2 ore in 7 minuti, 4 nodi per 2 ore in 29 minuti;
* due job da 4 nodi insieme (16 + 16 GPU) partiti 98 s dopo l'invio;
* nodi interi da 8-24 ore e da 2, 4 o 8 nodi: partiti dopo 105-114 minuti.

## 4 novembre 2026

**Mattina, la coda piena.** 7.025 job in attesa su `boost_usr_prod`, nessun
nodo libero (3.054 allocati, 207 misti). La nostra priorità (144.400) era
sopra il 90° percentile della coda (143.054), eppure i job da 2-3 GPU,
anche di un'ora, non avevano una partenza prevista. Quelli da 1 GPU fino a
4 ore partivano in 16-24 s. A decidere è la durata del job più che la
priorità: lo scheduler fa entrare negli spazi liberi solo i job che
finiscono prima della partenza già riservata ai job davanti (backfill). Nel
corso della giornata i job da 3 GPU sono comunque partiti, dopo ore
(`sacct`: 7:58, 14:07, 16:04, 18:01).

**Pomeriggio, il fermo.** Login e storage non rispondono (avviso HPC-News
di CINECA "Leonardo: login and storage access disruption"):
* `git`, `ls` e `tail` su `/leonardo_work` restano bloccati e non
  rispondono a Ctrl-C;
* una nuova sessione ssh, anche con `bash --noprofile --norc`, non arriva al
  prompt, perché si ferma sulla `$HOME`;
* dopo il ritorno, `git pull` fallisce una volta per una libreria di
  `/leonardo/prod` non trovata;
* alle 18:01, alla ripartenza, tutte le nostre prove in attesa da 3 GPU
  (2-8 ore) partono insieme.

**Sera, dopo il ritorno.** Una nuova prova di forma alle 20:37
(`fastlane-probe/20261104_2037/`). Nei primi minuti partono solo i job da
30 minuti, anche su 16 GPU (la coppia 16 + 16), e quelli da 1 GPU; i job da
3 GPU o da nodi interi di 2 ore e oltre restano in attesa. Il rapporto a
+1, +6 e +24 ore dirà quanto hanno aspettato.

**I nodi di login, 21:18-21:23** (`top` su due nodi di login di Leonardo,
confrontati nello stesso momento con quelli di JUPITER e di LUMI):

| | Leonardo | JUPITER | LUMI |
|---|---|---|---|
| utenti sul nodo | 110-116 | 9 | 69 |
| carico (1 min) | 76-96 | 9,3 | 1,3 |
| CPU inattiva | 48-66% | 85,5% | 99,2% |
| CPU in attesa del disco (`wa`) | 16-27% | 0% | 0% |
| swap | 4 GB su 4, tutto usato | nessuno | nessuno |

Il carico di Leonardo non viene dalle CPU, che sono per metà ferme, ma dai
processi in attesa dello storage (stato `D`), che nel carico Linux contano.
Sul nodo giravano anche lavori che di solito si fanno altrove:
* il server di un gestore di flussi di lavoro, con decine di processi
  Python brevi lanciati di continuo che leggono e scrivono sul `$WORK`, e
  un processo di controllo che chiama `ps` ogni pochi secondi;
* server remoti di editor con il loro analizzatore di codice;
* sincronizzazioni di log di addestramento verso un servizio esterno;
* il calcolo del checksum di un modello intero.

Alle 21:28, il carico di ogni nodo di login (`uptime`, dal nostro PC):

| login01 | login02 | login05 | login07 |
|---:|---:|---:|---:|
| 31,5 | 71,4 | non risponde | 29,6 |

Il nodo dove gira il server del gestore di flussi di lavoro ha più del
doppio del carico degli altri due che rispondono. Alle 21:33 login07 aveva
88 utenti, carico 20,8, CPU inattiva all'83% e attesa del disco allo 0,1%:
lo storage a quell'ora rispondeva, e il 16-27% di `wa` di login02 era di
quel nodo, non di tutta la macchina. Per lavorare si sceglie
il login meno carico (`ssh <utente>@login07-ext.leonardo.cineca.it`)
invece del nome generico, che assegna un nodo a caso.

I thread di rete e di scarico verso Lustre (`kiblnd_*`,
`flush-lhomefs`, `flush-larchive`) erano tra i primi per CPU. Per noi
l'effetto è un login lento a ogni comando che tocca il disco. Il calcolo
non ne risente: la stessa sera le prove di A2-mini sui nodi GPU andavano
alla velocità attesa (4.170 token/s su un'A100).

## Cosa ci è costato

Da tenere aggiornato con `sacct` e con le serie sopra:
* ore GPU perse per job morti a causa dell'infrastruttura: (al 4/11: nessuna
  misurata; i job morti quel giorno avevano errori nostri, cioè le catene
  A1 e il file sbagliato dell'esame di Flash-Next);
* ore di lavoro ferme per il fermo del 4/11: dal pomeriggio alle 18 circa;
* attese in coda per forma di job: dai rapporti delle prove e da
  `queue-stats`.
