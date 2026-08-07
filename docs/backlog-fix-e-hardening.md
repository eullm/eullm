# Backlog fix e hardening

**Baseline:** Engine v0.6.34 · Hub v0.1.0 · Forge v0.x · llama.cpp pinnato `9e3b928` · luglio 2026

Documento operativo complementare a [`roadmap-engine-0.7-1.0.md`](roadmap-engine-0.7-1.0.md)
(feature dello scheduler e delle API) e a [`forge-research-roadmap.md`](forge-research-roadmap.md)
(ricetta di distillazione e harness di valutazione). Quelle due roadmap coprono **cosa
costruire**; questa copre **cosa sistemare in ciò che già esiste**.

Ogni voce ha una checkbox — sostituire `[ ]` con `[x]` al completamento. Vale la
Definition of Done di `roadmap-engine-0.7-1.0.md`. Le voci già coperte dalle roadmap
esistenti non sono ripetute qui: sono elencate nella tabella dei rimandi in fondo.

---

## Principi

- **Nessuna voce senza riferimento al sorgente.** Ogni item indica file e riga così che sia
  verificabile prima di essere aperto e dopo essere chiuso.
- **Un fix per branch.** Vale in particolare per `scheduler.rs`, dove la roadmap 0.7→1.0
  impone già un cantiere per volta.
- **Prima la correttezza, poi la superficie.** Un componente che perde silenziosamente dati
  (audit, token, adapter) va sistemato prima di aggiungergli capacità.
- **I default devono essere quelli sicuri.** Dove un percorso opzionale è quello corretto,
  invertire il default invece di documentare il workaround.

---

## Stato

**Chiuse (v0.6.35):** tutte le H0, più H1-B, H1-D, H1-G, H2-A, H2-B, H2-C,
H2-G, H3-D, H3-K, H3-L, H3-M e D-01 — 18 voci su 36.

**Chiusa in v0.6.37:** H2-R (guardia sui logit NaN) — 28 su 46. La voce in più è
H2-S, aperta: il runner di build di `eullm-macos-x64` (vedi H2-S per cosa la
ricerca a monte ha escluso e cosa no).

**28 luglio (0.6.42 → 0.6.48), sette release in un giorno:** H2-V, H2-W, H2-X,
H2-Y, H2-Z, H2-AA, H2-AB, H2-AC, H2-AD, H3-O, H3-P, H3-Q. Undici voci su
dodici vengono da **due tester esterni o dall'uso diretto del prodotto**, non
da revisioni del codice: la chat che non riconosceva il modello caricato, gli
elenchi che ignoravano il disco, il proiettore raggiungibile solo dal catalogo
e la build da sorgente rotta da tre settimane erano tutte invisibili a 201
test verdi. Vale la pena registrarlo qui: in questa fase il rapporto costo/resa
migliore non ce l'ha una revisione a tavolino, ce l'ha far girare il prodotto e
leggere l'output di avvio di chi lo usa.

**In v0.6.39:** H2-T (Metal acceso su un binario che dichiarava di non averlo —
la voce più importante del documento) e H2-J (`think: false` non sopprimeva
niente). Entrambe trovate su segnalazioni di Peter, nessuna delle due da una
revisione a tavolino.

**In v0.6.38:** solo il cambio di runner di H2-S. Nessuna riga di codice del
motore: il binario `eullm-macos-x64` è lo stesso sorgente della 0.6.37
compilato su hardware Intel nativo invece che cross-compilato. È proprio questo
che la rende una prova utile — se le due macchine Intel di Peter cambiano
comportamento, la differenza è l'ambiente di build e nient'altro.

**Chiuse in v0.6.36:** H1-A, H1-C, H1-E, H2-K e H2-L — 22 voci chiuse su 39.
**Chiuse dopo la 0.6.36:** H2-M (default dei thread), H2-N (contesto per slot)
H2-O (`done_reason`), H2-P (`--daemon`) e H2-Q (audit concorrente) — 27 su 44. Le ultime due non vengono da una revisione del
codice ma dai report di un tester esterno sull'issue #140: vale la pena
notarlo, perché sono anche le due con l'impatto più diretto su chi usa il
prodotto. Il gate di uscita della tier H1 è soddisfatto **per l'Engine**: un
deployment in container ha ora un controllo d'accesso funzionante *ed
esprimibile*, e ogni input esterno che diventa un percorso o una richiesta di
rete è validato. Resta **H1-F**, che è lo stesso gate applicato all'**Hub**: il
suo listener è ancora `0.0.0.0` senza nessuno di questi controlli. Ora è
sbloccata (dipendeva da H1-A) ma non è un cambiamento dello stesso tipo — vuole
l'estrazione delle tre policy in un crate condiviso del workspace, quindi tocca
due binari e la struttura dei moduli. Tenuta fuori da questo blocco per non
mescolare un refactor di workspace con l'hardening dell'Engine.
Engine 174 test (da 114), clippy pulito con `-D warnings`, `cargo fmt` pulito.

La cosa che tiene insieme le tre voci è una regola di configurazione, ora scritta
in `CLAUDE.md`: **la configurazione di modello e inferenza va nei flag CLI, quella
di perimetro e policy nelle variabili d'ambiente.** Non è una preferenza di stile.
Un segreto su riga di comando è leggibile in `ps` da ogni utente locale; ogni
deployment non interattivo configura l'ambiente e non argv; e un'impostazione di
perimetro aggiunta come flag CLI andrebbe aggiunta a `run` *e* a `serve` e
cablata in `ServeConfig`, che è esattamente il meccanismo che ha prodotto la
divergenza `cache_type_k`/`gpu_layers` del luglio 2026. Una variabile letta
dentro `api::serve` è letta una volta, da entrambi i comandi, e non può
divergere.

Lo smoke test ha una sezione **perimetro** che avvia un secondo engine con le
chiavi configurate e verifica: ammissione decisa dal token (6 casi, incluso l'id
della chiave usato come token, che sarebbe un bypass totale), la sfida
`WWW-Authenticate`, il token da query string rifiutato sull'API, la quota per
chiave con `Retry-After`, che la quota sia per chiave e non globale, il rifiuto
cross-origin sui metodi con effetti collaterali (4 casi), l'assenza di oracolo
sul filesystem, che nessun segreto compaia nei log, e un record di audit con
`user_id` valorizzato. Il run principale lascia deliberatamente le chiavi
*non* configurate, così la postura di default resta quella misurata da tutto il
resto: 31 PASS, 0 FAIL, 3 SKIP in locale.

Engine 114 test, Hub 3, Forge 136 — tutti verdi; clippy pulito con
`-D warnings`; ruff pulito su `eullm_forge/`. I nuovi test coprono: limiti
degli override da body (7), precedenza di `EULLM_AUDIT_DIR` e scrivibilità (3),
precedenza ambiente/file per l'allowlist (5), sicurezza dei nomi file esterni
(4), `attention.key_length` nel dimensionamento KV (5), ordine degli stadi
della pipeline e contratto GGUF (5), default e contratto del record di
anonimizzazione (5).

**Non facciamo il trigger `pull_request` sulla CI** (era in H3-A). La CI gira
solo su `push: [main]` e impiega ~7 minuti stabili su 30 run consecutivi, tutte
verdi: aggiungere la validazione pre-merge raddoppierebbe i run per ogni
modifica in cambio di un'assicurazione contro un evento che non si è mai
verificato. Il buco reale è un altro e si chiude senza run aggiuntivi — vedi
H3-M.

H3-M è agganciata al solo job `release`, non ai sei job di build: compilare da
un commit non verificato non costa nulla che conti (runner standard su repo
pubblico), mentre *pubblicare* da lì è il danno. Tenere i build invariati
mantiene la modifica leggibile nel workflow che storicamente è il posto più
costoso in cui sbagliare. È anche l'unica modifica di questo blocco che non è
verificabile in locale: la logica di polling è stata simulata con uno stub di
`gh` sui cinque esiti (verde, rosso, run assente, in corso→verde, timeout).

**Verifica end-to-end sul binario pubblicato v0.6.35** (`eullm-linux-x64`,
checksum confrontato con `checksums.txt` e con il digest dell'API):
`eullm -V` riporta `0.6.35` — il version string non è rimasto indietro come in
0.6.32; `EULLM_ALLOWED_IPS` dall'ambiente compare nel log di avvio con la
sorgente corretta e il loopback conservato (H1-B); l'audit trail viene creato in
`EULLM_AUDIT_DIR` e ha registrato 12 richieste reali (H0-B); i sette casi fuori
intervallo di `batch_size`/`ctx_size` danno 400 con messaggio esplicito e i
valori validi passano (H0-A); la chat UI serve `Content-Security-Policy`,
`X-Content-Type-Options` e `Referrer-Policy` (H3-L). Inferenza reale con
qwen3-0.6b: non-streaming, NDJSON su `/api/chat` e SSE su
`/v1/chat/completions` tutti corretti, e **8 richieste concorrenti su 4 slot
hanno risposto ognuna alla propria domanda** (2→20 … 9→90), che è l'invariante
di H2-C sotto concorrenza reale. Il pull ha esercitato anche il download a range
paralleli e la verifica SHA-256 contro il digest del catalogo. Due scostamenti trovati e registrati: H2-I (404 invece di 500) e H3-N (banner
assente su `serve`), il secondo emerso scrivendo l'harness.

**La stessa verifica su ARM64** (Radxa Orion O6, Ubuntu 6.14 aarch64, binario
CIX-P1 CPU-only, `sha256 05611181…`): 19 PASS, 0 FAIL, 3 SKIP. Contano tre cose.
Le otto richieste concorrenti hanno risposto ognuna alla propria domanda anche
qui, quindi l'invariante di H2-C non dipendeva da un dettaglio di x86. Il banner
conferma che il build CPU sfrutta l'ISA del CIX-P1
(`NEON | ARM_FMA | FP16_VA | MATMUL_INT8 | SVE | DOTPROD | SVE_CNT = 16 | OPENMP | REPACK`),
cioè che la porta ARM non sta girando in modalità scalare. E i tre SKIP sono
tutti attesi e già tracciati: H2-I, H1-A (`user_id` sempre `null` perché non
esiste ancora un'identità di richiesta) e l'assenza di GPU su un binario CPU.
Un difetto nuovo è emerso da questo run e non era visibile su x86: H2-J.

**E su Linux CUDA** (Ryzen 9 5950X, RTX 5070 Ti 16 GB, driver 595.84, binario
`eullm-linux-x64-cuda-12.8`): 22 PASS, 0 FAIL, 2 SKIP — gli unici due sono H2-I
e H1-A, cioè voci aperte e non regressioni. Il banner conferma il build Blackwell
(`CUDA : ARCHS = 860,890,1200 | USE_GRAPHS = 1 | BLACKWELL_NATIVE_FP4 = 1`) e su
GPU le 8 richieste concorrenti chiudono in 0,2 s contro 1,5 s su ARM, tutte
corrette. Questo è anche l'unico dei tre run in cui il ramo di backpressure di
H2-B è stato plausibilmente raggiunto (500 byte trasferiti integralmente a un
client lento, contro i 12 eventi del run ARM che non arrivano alla capacità di
256 del canale).

**H2-A verificata su hardware, non solo dai test unitari.** Il primo run CUDA
non dimostrava niente sul dimensionamento della KV cache: a `--ctx-size 32768`
`--fit` rispondeva `model (2.33 GiB) fits fully in 14.65 GiB free VRAM`, che è
la conclusione a cui arriva *anche* l'aritmetica che sottostimava. L'errore era
per token e per layer, quindi cambia la decisione solo quando la cache — non i
pesi — è ciò che riempie la scheda. Rifatto con `--ctx-size 98304`:

```
[EULLM] --fit: model does not fit fully (14.67 GiB free VRAM, model 2.33 GiB).
        Offloading 30/36 layers, rest in RAM
```

e il modello ha poi servito una richiesta (HTTP 200). I numeri tornano
esattamente: `usable` = 14,67 × 0,97 − 640 MiB = 13,605 GiB; pesi per layer
0,0647 GiB; KV per layer a `key_length` 128 = 98304 × 8 × 128 × 2 × 2 B =
0,375 GiB → ⌊13,605 / 0,4397⌋ = **30 layer su 36**, che è ciò che l'engine ha
stampato. Con la vecchia assunzione `n_embd / n_head` = 2560/32 = 80 la KV per
layer sarebbe stata 0,234 GiB → ⌊13,605 / 0,2991⌋ = 45 ≥ 36 → **`FitsFully`**,
cioè offload di tutti i 36 layer, che a `head_dim` 128 reale richiedono 15,83 GiB
contro 14,67 liberi: OOM in allocazione. È esattamente il fallimento che `--fit`
esiste per evitare, e questo run è la prova che ora non si verifica.

Da qui una regola di metodo, perché il primo run CUDA era un falso verde:
un controllo che passa sia con il bug sia senza non è una verifica. Per questo
`tools/smoke_test.py` ha ora `--fit-ctx` e **dichiara nel report in quale regime
è finito**, etichettando un `fits fully` come non discriminante invece di
lasciarlo leggere come conferma.

**Correzione a H0-B fatta prima della release.** Il controllo all'avvio era
troppo aggressivo: rifiutava di partire ogni volta che la directory di audit non
era scrivibile, quindi una home in sola lettura diventava un'interruzione di
servizio per un file di log che nessuno aveva chiesto. Ora l'errore è fatale
**solo** se `EULLM_AUDIT_DIR` è impostata — chi la imposta, o monta un volume su
di essa, ha dichiarato che il registro conta; altrimenti si avvisa e si serve.
La postura severa è una scelta dell'operatore, non un default da imporre.

Prossimo blocco consigliato: **H2-F** (guardie sul patcher GGUF), poi **H2-D**
(stop sequence e filtri unificati fra i due percorsi di inferenza, che è anche
dove va la guardia di H2-J), poi **H3-C** (`cargo deny` e `pip-audit` in CI —
tenuta deliberatamente fuori dalla 0.6.35 perché può far fallire il build al
primo advisory trovato, e non è il momento di scoprirlo mentre si taggia).

---

## H0 — Bloccanti  ·  ✅ chiuse

**Gate di uscita:** nessun componente perde silenziosamente dati che dichiara di gestire;
nessun campo ricevuto dalla rete raggiunge una decisione di allocazione senza limiti.

- [x] **H0-A · Limiti sugli override ricevuti nel body delle richieste** *(P0)*
  `batch_size` e `ctx_size` arrivano dal body a `swap_model` senza controllo di intervallo
  (`api/routes.rs:513-520, 656-663, 941-948` → `api/mod.rs:197, 220` →
  `inference/scheduler.rs:679, 793`). Valori fuori scala portano lo scheduler in uno stato
  in cui nessuna richiesta è più servibile, oppure fanno fallire l'allocazione del contesto
  *dopo* che il modello precedente è già stato scaricato (`api/mod.rs:171`), lasciando lo
  slot vuoto. Clamp a intervalli sensati (`batch_size` 1..=64, `ctx_size` 512..=limite
  configurato) e HTTP 400 esplicito fuori range. Il `CLAUDE.md` del progetto prescrive già
  di validare ogni override che arriva da un body: qui il pattern
  `override_*.unwrap_or(self.*)` è stato applicato senza la validazione che lo accompagna.
  Sinergia con `0.7-C` (backpressure e codici di errore), da cui questa voce è la
  specializzazione concreta sui due campi che oggi non hanno limiti.

- [x] **H0-B · L'audit trail deve rispettare `EULLM_AUDIT_DIR`** *(P0)*
  `engine/Dockerfile:47` imposta `ENV EULLM_AUDIT_DIR=/data/audit` e `docker-compose.yml`
  monta un volume su quel percorso, ma **la variabile non è letta da nessuna parte nel
  codice**: `AuditLogger::default_path()` (`audit/mod.rs:88-96`) usa esclusivamente
  `HOME`/`USERPROFILE`. In ogni deployment containerizzato il registro finisce nel layer
  effimero del container e si perde alla ricreazione, mentre il volume montato resta vuoto.
  Leggere la variabile come già si fa per `EULLM_MODELS_DIR` (`models/store.rs:57`) e
  rifiutare l'avvio se la directory non è scrivibile. Una riga per il fix, più il controllo
  all'avvio.

- [x] **H0-C · L'adapter di identità deve arrivare nel GGUF finale** *(P0)*
  In `forge/eullm_forge/pipeline.py:158-176` lo stadio 4 assegna `adapter_path` ma non
  aggiorna `current_model_path` né fonde l'adapter nei pesi; lo stadio 5 esporta quindi il
  modello **pre-LoRA**. `eullm forge --identity "…"` completa senza errori e produce un
  GGUF privo dell'identità richiesta, dopo aver speso il tempo GPU dello stadio 4.
  Nota: `forge-research-roadmap.md` §F2 elenca l'identity LoRA come `[✅ implementato]` —
  il modulo lo è, l'orchestrazione che lo collega no. Fondere l'adapter
  (`merge_and_unload()`), salvare il checkpoint fuso, assegnarlo a `current_model_path`,
  e aggiungere un test che verifichi che il path esportato discende dallo stadio 4 quando
  `skip_identity` è falso.

- [x] **H0-D · Riordinare la pipeline per il target GGUF** *(P0, insieme a H0-C)*
  Lo stadio 3 di `pipeline.py` produce un checkpoint AWQ/GPTQ (`quantize.py:132-170`;
  `profiles/legal_it.yaml:24` specifica `method: awq`) e lo stadio 5 gli passa
  `convert_hf_to_gguf.py --outtype f16` (`export.py:204-212`), che legge tensori fp16/bf16 e
  non pesi impacchettati a 4 bit: il profilo documentato non completa end-to-end. È anche
  una doppia quantizzazione concettuale — nella documentazione di architettura
  "quantizzazione" indica il passaggio `FP16 → Q4_K_M`, cioè esattamente ciò che lo stadio 5
  fa già con `llama-quantize`. Ordine corretto per un target GGUF:
  pruning → distillazione → identity LoRA fusa → export GGUF F16 → `llama-quantize` Q4_K_M.
  Togliere AWQ/GPTQ dal percorso GGUF conservando `quantize.py` come utilità separata per
  chi esporta verso vLLM/TensorRT; aggiornare i tre profili e la documentazione della
  pipeline, che oggi descrive l'ordine sbagliato. Da coordinare con la decisione sulla
  ricetta canonica già aperta in `forge-research-roadmap.md` §"To fix / reconcile".

- [x] **H0-E · NER attivo per default nell'anonimizzazione** *(P0)*
  `AnonymiserConfig.use_ner` è `False` (`datasets/anonymize.py:190`) e
  `scripts/anonymize_italgiure.py:122-125, 164` lo attiva solo con `--ner`. Nella
  configurazione predefinita l'unico livello che redige nomi di persona è
  `RE_ALLCAPS_NAME`, che per costruzione richiede token interamente maiuscoli
  (`anonymize.py:123-127`): un nome in Title Case — la forma in cui compare nel corpo
  motivazionale, non nell'intestazione OCR — non viene toccato da nessun livello. Invertire
  il default (`--no-ner` per disattivare) e uscire con errore, non degradare in silenzio, se
  spaCy o `it_core_news_lg` non sono disponibili quando il NER è richiesto. Comporta
  promuovere `it_core_news_lg` a dipendenza non opzionale del percorso legale.
  Il principio vincolante "GDPR-safe dataset: anonymized corpus only" di
  `forge-research-roadmap.md` non è oggi garantito dal default.

- [x] **H0-F · Allineare il claim di anonimizzazione e rimuovere il riferimento alla fonte** *(P0)*
  Il docstring di `anonymize.py:33` afferma «the redaction is one-way: original text is not
  recoverable from the output». Non è così: `anonymize_record` (`:589-597`) sostituisce solo
  il campo `text`, mentre il record prodotto da `datasets/italgiure.py:190-208` conserva
  `url` (link diretto al PDF originale non redatto), `sentence_id`, `article_num` e
  `metadata.ecli`, ciascuno dei quali identifica univocamente la sentenza. Il risultato è
  **pseudonimizzazione** ai sensi dell'art. 4(5) e del considerando 26 GDPR, non
  anonimizzazione, e resta quindi nell'ambito di applicazione del Regolamento. La
  mitigazione del rischio di memorizzazione nei pesi resta valida e utile: è l'etichetta
  giuridica a essere sbagliata. Rimuovere `url` dai record destinati al training (o
  sostituirlo con un identificativo con hash e salt tenuto separato dal corpus), correggere
  il docstring e documentare il rischio residuo di re-identificazione. Da chiudere **prima**
  di qualunque pubblicazione del corpus, e prerequisito della sonda di membership inference
  già prevista in `forge-research-roadmap.md` §F2.

---

## H1 — Perimetro e controllo d'accesso

**Gate di uscita:** un deployment in container ha un controllo d'accesso funzionante ed
esprimibile; ogni input esterno che diventa un percorso o una richiesta di rete è validato.

- [x] **H1-A · Autenticazione opzionale a token con quote per chiave** *(P1)*
  Oggi l'unico controllo è l'allowlist IP (`api/ip_allowlist.rs`, middleware più esterno in
  `api/mod.rs:604-607, 628-631`), che è la scelta giusta per l'uso locale ma non è
  esprimibile dietro il port publishing di Docker: con `ports: "11434:11434"`
  (`docker-compose.yml:18-19, 32-33`) il traffico esterno arriva al processo con l'IP del
  gateway del bridge, mai con il loopback, quindi l'operatore o non riesce a usare il
  servizio o allarga l'allowlist alla subnet — e poiché ogni client esterno viene tradotto
  in quell'unico indirizzo, l'allargamento non discrimina più nulla. Un bearer token
  opzionale verificato **prima** dell'allowlist, con quote e rate limit per chiave e
  l'identificativo della chiave propagato in ogni riga di audit, è l'unico controllo che
  sopravvive alla traduzione degli indirizzi. Fornisce anche il soggetto che `user_id`
  (`audit/mod.rs:36`) oggi non ha mai. Precondizione di ogni deployment multi-tenant.
  **Chiuso in `api/auth.rs`** (nuovo modulo, 20 test). Configurazione da
  `EULLM_API_KEYS`, poi `EULLM_API_KEYS_FILE`, poi `.env` — mai da flag CLI, per
  la regola scritta ora in `CLAUDE.md`: un segreto su riga di comando è leggibile
  in `ps` da qualunque utente locale. Formato `id:secret[:rpm=N]`; il segreto è
  tenuto solo come digest SHA-256 e confrontato a lunghezza fissa, così il
  confronto è a tempo costante per costruzione e un core dump non consegna il
  token. Tre decisioni da capire prima di modificarlo:
  1. **Il middleware è più esterno dell'allowlist e una chiave valida la
     scavalca.** È l'unico ordinamento che risolve il caso Docker: rifiutare una
     chiave valida perché il pacchetto arriva dal gateway del bridge lascia
     l'operatore dove era. Abilitare le chiavi quindi *sostituisce* l'ammissione
     per indirizzo con quella per identità, non ci si somma. L'avvio lo scrive a
     log in chiaro, perché è il punto in cui abilitare un controllo ne allenta un
     altro.
  2. **Una configurazione che non parsa è fatale all'avvio.** Chi imposta
     `EULLM_API_KEYS` ha chiesto autenticazione; servire aperto per un errore di
     battitura è l'unico esito che non deve essere possibile. Stessa logica di
     H0-B su `EULLM_AUDIT_DIR`.
  3. **Il token da query string è accettato solo sul listener della UI.** Un
     browser non può impostare un header alla prima navigazione, quindi
     `?api_key=…` fa il bootstrap e la pagina lo mette in `sessionStorage` e lo
     manda come header (`ui/app.js`, `withAuth`); l'URL viene ripulito subito con
     `history.replaceState`. Sull'API è rifiutato: un token in una URL finisce
     nei log dei proxy, nella cronologia del browser e nell'header `Referer`.
  Quota per chiave a finestra fissa di un minuto → 429 con `Retry-After`. E
  `user_id` nell'audit ora è l'id della chiave: verificato su hardware, un record
  con `user_id='ci'` invece di `null`.

- [x] **H1-B · `EULLM_ALLOWED_IPS` anche dall'ambiente di processo** *(P1)*
  `IpAllowlist::load_from_env_file` è invocata con il percorso letterale `".env"` relativo
  alla working directory (`api/mod.rs:436`) e la variabile non è mai letta dall'ambiente.
  `docker run -e EULLM_ALLOWED_IPS=…` e un `Environment=` in un'unità systemd non hanno
  alcun effetto, senza avviso; nessuna immagine del repository include un `.env`, quindi in
  container l'allowlist è sempre e solo loopback. Leggere la variabile dall'ambiente con
  precedenza sul file, e loggare all'avvio la sorgente effettiva della configurazione.

- [x] **H1-C · Hardening del web tool** *(P1)*
  `tools::fetch_url` (`tools/mod.rs:77-96`) scarica la URL estratta dal messaggio utente
  senza validazione dell'host, senza limiti sul corpo della risposta
  (`resp.text()`, `:94`) e seguendo i redirect con la policy di default di reqwest.
  Aggiungere: risoluzione dell'host con verifica che ogni indirizzo risultante sia pubblico
  (escludere loopback, RFC1918, link-local, ULA IPv6), ripetizione del controllo su ogni hop
  di redirect o disabilitazione dei redirect, limite di byte leggendo a chunk invece che con
  `text()`, schema `https` per default con opt-in esplicito per `http`, e una allowlist di
  domini configurabile — che per il caso d'uso enterprise è la forma più difendibile della
  feature.
  **Chiuso in `tools/guard.rs`** (nuovo modulo, 14 test). `https` obbligatorio
  salvo `EULLM_WEB_ALLOW_HTTP=1`; l'host è risolto e **tutti** gli indirizzi
  risultanti devono essere pubblici, non solo il primo — un nome con un record
  pubblico e uno su `10.0.0.5` sarebbe altrimenti un lancio di dado dipendente
  dall'ordine, cioè il tipo di buco che non si trova mai; la connessione è poi
  fissata all'indirizzo verificato con `ClientBuilder::resolve`, che chiude il DNS
  rebinding fra controllo e connect; i redirect sono disabilitati nel client e
  seguiti a mano rifacendo *tutti* i controlli su ogni hop (validare la URL e poi
  lasciar seguire i redirect non valida nulla); corpo letto a chunk con cap a
  4 MiB; solo content type testuali, e un `Content-Type` assente è rifiutato
  invece di essere assunto testuale; `EULLM_WEB_ALLOWED_DOMAINS` limita alle
  fonti dichiarate, con match sui confini di label — `evil-example.com` non
  soddisfa `example.com`.
  Due dettagli coperti dai test perché sono i bypass classici: le forme IPv6 che
  incapsulano un IPv4 (`::ffff:169.254.169.254`, NAT64 `64:ff9b::/96`, 6to4
  `2002::/16`) sono giudicate sull'indirizzo che trasportano, non sulla
  rappresentazione; e `0.0.0.0/8` è fra i non pubblici, il che è corretto di per
  sé (RFC 6890) *e* impedisce che `::1` — che vive dentro il range
  IPv4-compatible `::/96` — venga scartato a `0.0.0.1` e giudicato pubblico.
  Questo secondo caso era un bug reale nella mia prima stesura, trovato dai test
  e non dalla lettura.

- [x] **H1-D · Validare i nomi file restituiti dall'API HuggingFace** *(P1)*
  `list_hf_ggufs` raccoglie i `siblings[].rfilename` filtrandoli solo per il suffisso
  `.gguf` (`registry/mod.rs:543-551`), e `cmd_pull_hf` li usa direttamente come componente
  di percorso: `model_dir.join(&filename)` (`main.rs:920`). Un nome contenente separatori,
  `..` o un percorso assoluto non resta dentro la directory del modello. Rifiutare ogni
  `rfilename` che non sia un singolo componente sicuro, riusando la logica di
  `is_valid_model_slug` che esiste già in `hub/src/main.rs:356-365` invece di riscriverla.
  Applicare lo stesso controllo ai campi `gguf_file`/`mmproj_file` letti dal manifest
  (`models/store.rs:159, 193`).

- [x] **H1-E · Restringere l'origine CORS e i percorsi modello accettati** *(P1)*
  Due voci che condividono la stessa causa — il perimetro assume un chiamante fidato.
  (a) CORS è `allow_origin(Any)` + `allow_methods(Any)` + `allow_headers(Any)`
  (`api/mod.rs:594-597, 617-620`): quando la richiesta parte dal browser dell'utente l'IP
  del socket *è* il loopback, quindi l'allowlist è superata per costruzione e qualsiasi
  pagina visitata può leggere le risposte degli endpoint, inclusi swap e unload del modello.
  Restringere l'origine agli host della UI e verificare `Origin` sulle richieste con effetti
  collaterali. (b) `resolve_model` restituisce il percorso così com'è se `path.is_file()`
  (`api/mod.rs:329-331`), quindi il campo `model` di una richiesta può indicare qualunque
  file locale, e i messaggi d'errore propagati al client (`routes.rs:91-96`) distinguono i
  casi. Consentire percorsi arbitrari solo dietro un flag di avvio esplicito e uniformare il
  messaggio d'errore.
  **Chiuso.** (a) `api/origin.rs` (nuovo modulo, 9 test) con
  `EULLM_ALLOWED_ORIGINS`; il default consente qualunque origine di loopback su
  qualunque porta — così la chat UI e un frontend locale continuano a funzionare
  — e rifiuta tutto il resto. Il punto non ovvio: **CORS da solo non è il
  controllo.** CORS decide se il browser restituisce la *risposta* alla pagina; la
  richiesta viene comunque eseguita, e un `POST` con `Content-Type: text/plain`
  non fa nemmeno preflight. Quindi la policy è usata due volte: come predicato
  allow-origin del `CorsLayer`, e come `enforce_origin` che **rifiuta con 403**
  ogni metodo non sicuro con `Origin` non consentita, prima dell'handler. Le
  richieste senza header `Origin` restano intoccate — sono tutti i client non
  browser, e romperle costerebbe compatibilità senza guadagnare niente, dato che
  un programma può mandare gli header che vuole.
  (b) `resolve_model` accetta percorsi arbitrari solo con
  `EULLM_ALLOW_MODEL_PATHS=1`; senza, risolve nel model store e nei mount
  deliberati (`/models`, `/data/models`, e solo con un nome file sicuro joinato,
  altrimenti `../` uscirebbe subito), più il modello di lancio per nome o per
  percorso esatto — mai per stem o prefisso — perché `/api/tags` annuncia quel
  nome e rifiutare la propria risposta romperebbe `eullm run ./model.gguf` al
  primo swap di ritorno. L'oracolo sul filesystem è chiuso: verificato che
  `/etc/hostname` (esiste) e `/etc/eullm-definitely-not-here` (non esiste)
  producono ora un errore identico a meno del nome che il chiamante ha fornito
  lui stesso, mentre con il flag attivo il primo arriva al loader e dà un errore
  diverso. Lo smoke test fa esattamente questo diff a ogni release.

- [ ] **H1-F · Allineare l'Hub al perimetro dell'Engine** *(P1)*
  `hub/src/main.rs:65-69` ascolta su `0.0.0.0` senza allowlist, autenticazione, CORS né rate
  limit, a differenza dell'Engine. Riusare `ip_allowlist` (estraendolo in un modulo
  condiviso del workspace) e applicare H1-A quando disponibile. Correggere nello stesso
  passaggio la configurazione Docker: `hub/Dockerfile:38` dichiara `EXPOSE 3000` e il
  compose mappa `3000:3000`, ma il binario ascolta su 8080 in assenza di `EULLM_HUB_PORT`
  (`hub/src/main.rs:61-64`) — il servizio così com'è non è raggiungibile.

- [x] **H1-G · Le card dell'Hub devono descrivere modelli esistenti** *(P1)*
  `model_card` e `compliance_card` (`hub/src/main.rs:174-254`) non consultano il catalogo e
  restituiscono 200 con contenuto affermativo per **qualsiasi** nome, inclusi modelli che
  non esistono, con asserzioni come `"gdpr_compliant": true` e `"personal_data": "No
  personal data in training set"`. Per un progetto la cui proposta di valore è la conformità
  documentata, una card di conformità generata per un modello inesistente è il problema di
  integrità più serio della superficie pubblica. Restituire 404 per i nomi non in catalogo,
  e in prospettiva servire la card **generata da Forge** per quel modello specifico
  (`forge-research-roadmap.md` §F2, "Per-model compliance card generated by Forge", che nota
  già che «today the Hub serves a static stub») invece di un testo statico.

---

## H2 — Correttezza e robustezza

**Gate di uscita:** nessun percorso di generazione consegna al client meno di quanto ha
prodotto; i due percorsi di inferenza si comportano allo stesso modo; i parser di formati
esterni non si fidano dei valori che leggono.

- [x] **H2-A · Dimensionare la KV cache con `attention.key_length`** *(P1)*
  Sia il sizer di `--fit` (`fit.rs:58-64`) sia la stima runtime (`scheduler.rs:1657`)
  calcolano `head_dim = n_embd / n_head`, ignorando le chiavi GGUF
  `<arch>.attention.key_length` / `.value_length` che molte architetture dichiarano
  esplicitamente. Su Qwen3-4B (n_embd 2560, n_head 32) il calcolo dà 80 contro un
  `head_dim` reale di 128: la KV è sottostimata del 37%, `--fit` offloada più layer di
  quanti entrino e il caricamento fallisce per OOM — su un modello presente nel catalogo.
  Leggere le due chiavi nel parser (che è già bounds-checked e facile da estendere) e usarle
  quando presenti, con l'attuale formula come fallback.
  **Verificata su RTX 5070 Ti** a `--ctx-size 98304`: 30/36 layer offloadati e
  richiesta servita, dove la vecchia aritmetica avrebbe dichiarato un fit pieno
  da 15,83 GiB su 14,67 liberi. Aritmetica completa nella sezione Stato.

- [x] **H2-B · Backpressure invece di scarto silenzioso dei token** *(P1)*
  `send_or_detect_disconnect` (`scheduler.rs:1734-1739`) tratta correttamente un canale
  pieno come "client lento, non disconnesso", ma **scarta il token** e prosegue: il client
  riceve testo con parti mancanti, senza alcun errore, e il conteggio finale non corrisponde
  a quanto consegnato. Applicare backpressure sulla sequenza (sospenderla per un'iterazione)
  oppure chiudere lo stream con un errore esplicito. Il canale ha capacità 256
  (`scheduler.rs:236`), quindi la condizione richiede un consumatore molto lento — ma è una
  perdita di dati silenziosa, non una degradazione.

- [x] **H2-C · Trattare come errore il fallimento di `decode_batch.add`** *(P1)*
  Alla fase 3 del loop un `add` fallito produce solo un `warn` e la sequenza non entra nel
  batch (`scheduler.rs:1236-1244`), ma la fase 5 le assegna comunque un `logit_idx`
  (`:1282-1304`): da quel punto i logit di tutte le sequenze successive sono disallineati e
  ciascuna campiona dalla distribuzione di un'altra conversazione. Oggi la condizione non
  dovrebbe verificarsi (il batch è dimensionato su `max_batch_size` e `active.len()` non lo
  supera), ma è un invariante non presidiato in un punto in cui la violazione è silenziosa e
  produce output plausibile. Contare esplicitamente le sequenze effettivamente aggiunte e
  derivare `logit_idx` da quel conteggio, oppure abortire l'iterazione.

- [x] **H2-D · Unificare stop sequence e filtri tra i due percorsi di inferenza** *(P1)*
  L'hold-back buffer e i `filter_sequences` esistono solo nello scheduler
  (`process_piece`, `scheduler.rs:1588-1628`). Il percorso sequenziale e quello multimodale
  (`inference/mod.rs:1047-1085, 1357-1387`) confrontano `full_output.ends_with(s)` per
  token, quindi una stop sequence a cavallo di due token perde testo, e
  `DEFAULT_HARMONY_FILTERS` non è applicato affatto: gli artefatti `<|channel>…` che i
  filtri esistono per rimuovere restano visibili all'utente in modalità multimodale.
  Estrarre `process_piece` in un helper condiviso e usarlo in tutti e tre i loop di decode.
  Sinergia con H2-E: entrambe le voci sono conseguenze della stessa duplicazione.

  **Chiusa.** `inference/output.rs` contiene ora l'unica implementazione, che è
  quella dello scheduler spostata **senza modifiche di comportamento** — il
  codice già collaudato in produzione, non una riscrittura. I tre loop la
  chiamano: scheduler, motore sequenziale (non streaming e streaming) e
  multimodale.

  Scrivendo i test è emerso che i difetti erano tre e non uno, e il terzo era
  peggiore di come l'avevamo descritto:
  1. **Una stop sequence che non finisce il pezzo veniva ignorata.** Un pezzo
     `"<|im_end|>\n"` lascia l'output accumulato che finisce con `"\n"`, quindi
     `ends_with` non vedeva niente e la generazione proseguiva oltre il turno.
  2. **A cavallo di due token trapelava la prima metà.** I percorsi in streaming
     ritagliavano il marcatore dal pezzo *corrente*, ma il pezzo precedente coi
     primi byte era già partito verso il client e non si richiama indietro.
  3. **Quel ritaglio era un indice di byte su una stringa UTF-8**
     (`&piece[..piece.len() - s.len()]`): con un carattere multibyte a cavallo
     del taglio non sbagliava, **andava in panic**. Non c'era nel testo della
     voce; è saltato fuori scrivendo il test sui multibyte.

  Più il difetto già noto: `filter_sequences` non era applicato fuori dallo
  scheduler, quindi in multimodale gli artefatti `<|channel|>` andavano
  all'utente così com'erano.

  Aggiunto anche `flush()`, che rende esplicita una distinzione che prima era
  implicita e sbagliata negli altri due loop: su **EOG** la coda trattenuta è un
  delimitatore di turno parziale e si butta; a **budget esaurito** è testo vero
  e buttarla tronca la risposta. Lo scheduler la emetteva già; gli altri due la
  perdevano in silenzio.

  Verifica: 9 test nuovi che coprono i tre difetti uno per uno (184 totali),
  `--all-targets` con e senza `multimodal`, e smoke test sul binario in release
  su entrambi i percorsi, scheduler e `--batch-size 0`, 32 pass e 0 fail.
  Nota di metodo: `cargo check` da solo **non compila i test** e mi ha lasciato
  passare un import rotto nel modulo di test dello scheduler. Va usato
  `cargo check --all-targets`.

  **Quello che non è dentro**, per non farlo sembrare più completo di quanto è:
  H2-E (la catena di sampling, ancora duplicata in quattro punti) e il punto 2
  di H2-J (la guardia sul tag `</think>` orfano). Quest'ultimo ora avrebbe un
  posto solo dove stare, ma non è un filtro incondizionato: `</think>` è
  legittimo quando `think: true`, quindi va aggiunto ai `filter_sequences`
  della singola richiesta quando `think` è falso. È una decisione lato
  chiamante, non lato `process_piece`, e merita il suo cambio.

  **Punto 2 chiuso in 0.6.40.** `inference::default_filters(think)` costruisce
  la lista per richiesta: le Harmony sempre, più `<think>` e `</think>` solo
  quando `think` è falso. Lì il prompt porta già un blocco think chiuso e
  vuoto, quindi un tag **in uscita** è spurio per costruzione. Con
  `think: true` i tag sono output legittimo che la UI rende come sezione di
  ragionamento e non vanno toccati.
  È una guardia, non un parser: il meccanismo dei filtri lavora su
  sottostringhe letterali, quindi toglie i tag ma non può eliminare un blocco
  `<think>…</think>` con contenuto dentro. Sopprimere il ragionamento è
  compito del prefisso, che è la correzione vera; questa è la seconda linea.
  Tre test, incluso quello che verifica che i filtri Harmony sopravvivano in
  entrambe le modalità — la regressione che una riscrittura ingenua causerebbe.

- [x] **H2-E · Estrarre la costruzione della catena di sampling** *(P2)*
  La catena è duplicata in quattro punti (`scheduler.rs:929-966`;
  `inference/mod.rs:781-812, 1000-1031, 1313-1341`) e la divergenza è già iniziata: il
  fallback del seed differisce (`seq_id` contro `1234` hardcoded) e i filtri sono assenti in
  tre copie su quattro. Ogni nuovo sampler andrà aggiunto quattro volte. Estrarre
  `fn build_sampler(&LlamaModel, &GenerateRequest, seed) -> LlamaSampler`.

  **Chiusa.** `inference/sampling.rs` contiene `build_sampler`, e le quattro
  copie sono sparite. Le due divergenze previste dalla voce c'erano entrambe,
  ed erano di gravità diversa:
  - il **fallback del seed**: `seq_id` nello scheduler, `1234` fisso nei tre
    percorsi sequenziali. Conta solo se l'orologio di sistema precede l'epoca
    Unix, quindi niente di visibile — sono due risposte a una domanda sola, che
    è il modo in cui cominciano le divergenze che poi contano. Ora il chiamante
    passa il proprio, e lo scheduler passa lo slot: due richieste concorrenti
    senza seed non possono collassare sulla stessa sequenza nemmeno in quel caso.
  - la **copia multimodale ingoiava gli errori di grammatica**: `if let Ok(gs)`
    senza `else`, mentre le altre tre loggano «Grammar sampler init failed».
    Una richiesta con `format: "json"` la cui grammatica non compilava tornava
    testo libero **senza una riga da nessuna parte**. Questo è un difetto vero,
    non solo duplicazione.

  L'ordine della catena è ora documentato come contratto e non come stile:
  grammatica → penalties → top-k → top-p → min-p → temperatura → dist. La
  grammatica deve vedere la distribuzione intera prima che qualcuno la tronchi,
  e `dist` deve restare ultimo perché è ciò che estrae davvero il token.

  Verifica sul binario, non solo unit test: **stesso seed due volte → risposta
  identica**, **senza seed → risposte diverse** a temperatura 1.4. È il
  controllo discriminante per una modifica al sampling, perché passa solo se il
  seed viene davvero usato e davvero variato. Più smoke test 32 pass / 0 fail,
  186 test, clippy, fmt, `--all-targets` con e senza `multimodal`.

  Nota di metodo, seconda volta oggi: `cmd | tail; echo $?` legge lo stato di
  `tail`, non di `cmd`. Mi ha nascosto un fallimento di clippy
  (`field_reassign_with_default` in un test nuovo) che avevo già dichiarato
  verde. Con le pipe serve `PIPESTATUS` o `set -o pipefail`.

- [ ] **H2-F · Irrobustire il patcher GGUF** *(P2)*
  A differenza del parser di `fit.rs`, che è bounds-checked con rigore,
  `gguf_patch.rs` si fida dei valori che legge: una lunghezza `u64` presa dal file diventa
  direttamente una dimensione di allocazione (`:216-221`), `count * scalar_size` non ha
  controllo di overflow (`:104, :249`) e l'offset di seek passa per `n as i64` (`:224-227`).
  Inoltre `ALIGNMENT` è la costante 32 (`:22`) mentre GGUF definisce la chiave
  `general.alignment` che può sovrascriverla: su un file con allineamento diverso il
  ricalcolo del padding produce un GGUF **silenziosamente corrotto**. Allineare il parser
  alle stesse guardie di `fit.rs` (che è il modello da seguire, nello stesso repository) e
  leggere `general.alignment`.

- [x] **H2-G · Leggere l'header GGUF con `read_exact`** *(P2)*
  `read_gguf_info` (`fit.rs:230-234`) usa una singola `file.read()`, che non garantisce di
  riempire il buffer: su una lettura parziale il parse fallisce e `--fit` ricade
  silenziosamente su `--gpu-layers`, in modo non deterministico. Usare
  `take(8 MiB).read_to_end()`.

- [ ] **H2-H · Ricalibrare `COMPUTE_BUFFER_RESERVE_BYTES`** *(P2)*
  Il valore è stato raddoppiato a 640 MiB per stima lineare dopo l'introduzione di
  `n_ubatch = 1024`, e il commento nel codice dichiara esplicitamente che non è stato
  verificato contro una misura reale (`fit.rs:290-298`). Su GPU da 8–12 GB sono 1–3 layer
  offloadabili persi per un margine non calibrato. Misurare il compute buffer effettivo a
  `n_ubatch=1024` (riga di log del loader, o `nvidia-smi`) su almeno due modelli e fissare
  il valore sul dato. Da fare insieme a `0.7-E`, che tocca lo stesso sizer.

- [x] **H2-I · Un modello inesistente deve dare 404, non 500** *(P2)*
  Emerso testando il binario pubblicato di v0.6.35: una richiesta con
  `{"model":"non-esiste"}` restituisce **500** con `Failed to load model ...`
  (`api/routes.rs:171-180` — l'errore di `swap_model` è mappato su
  `INTERNAL_SERVER_ERROR` indipendentemente dalla causa). Chiedere un modello che
  non c'è è un errore del client, non del server: Ollama risponde 404, e un 5xx
  induce i client con retry automatico a riprovare una richiesta che non potrà
  mai riuscire. Distinguere "non trovato" (404) da un fallimento reale di
  caricamento — VRAM insufficiente, GGUF corrotto — che resta 500. Comporta far
  ritornare a `resolve_model` un errore tipizzato invece di una `String`.

  **Chiusa.** `ModelError { NotFound, LoadFailed }` in `api/mod.rs`, con
  `From<String>` che manda tutto ciò che non è esplicitamente una ricerca
  fallita su `LoadFailed` — così il default è il 500 e il 404 va chiesto,
  non il contrario. `resolve_model` e `swap_model` lo restituiscono, e
  `routes.rs` mappa i due casi. Un solo chiamante di `swap_model`, quindi il
  cambio di firma non si è propagato.
  Verificato sul binario: `{"model":"non-esiste"}` risponde **404** su
  `/api/chat` e su `/v1/chat/completions`, un modello vero resta **200**. Lo
  smoke test aveva questo controllo come `skip` tollerato: ora è
  un'asserzione dura e il totale passa da 32 a 33 pass.

- [x] **H2-J · Con `think: false` un `</think>` finisce nel testo visibile** *(P2)*
  Emerso dallo smoke test su ARM (Radxa Orion O6, qwen3-0.6b, `think: false`,
  `temperature: 0`): il contenuto ricostruito dallo stream NDJSON è
  `"</think>\n\nCount: one two three"` — il tag di chiusura arriva al client come
  testo dell'assistente. **Riprodotto byte per byte su x86 con backend CUDA**
  (RTX 5070 Ti, stesso modello, stesso prompt): identico. Quindi non dipende
  dall'architettura né dal backend, ed è deterministico a `temperature: 0` —
  che esclude una spiegazione basata sul sampling e punta al prompt.
  Due cause candidate, indipendenti, entrambe da verificare prima di toccare il
  codice:
  1. Il prefisso iniettato è `"<think>\n</think>\n\n"` (`chat_template.rs:89-94`),
     mentre il template ufficiale di Qwen3 per `enable_thinking=false` emette
     `<think>\n\n</think>\n\n` — una riga vuota fra i due tag. Una sequenza fuori
     distribuzione di un solo newline è sufficiente a far riemettere al modello il
     tag di chiusura. Il fix è di un carattere, ma tocca due test che asseriscono
     la stringa attuale (`chat_template.rs:247` e `:264`) e — attenzione — anche
     la ricostruzione della history in `main.rs:2927`, che deve restare
     byte-identica al prefisso iniettato, altrimenti si rompe il riuso del prefix
     KV (è il motivo per cui `think_suppression_prefix()` esiste come funzione
     unica invece di due letterali).
  2. Indipendentemente dalla causa, l'engine non ha nessuna guardia che tolga un
     blocco `<think>…</think>` — o un tag di chiusura orfano — dal contenuto
     visibile. `process_piece` ha già il buffer di hold-back per le stop sequence
     (`inference/scheduler.rs`): è il posto dove filtrare, non a valle nel client.
  Il sintomo è già riproducibile con `tools/smoke_test.py` su qualunque box: a
  temperatura zero un before/after è misurabile senza ambiguità.

  **Causa 1 confermata, non per ipotesi: letta dal template dentro il GGUF.**
  `tokenizer.chat_template` di `Qwen3-0.6B-Q4_K_M.gguf` — lo stesso file su cui
  girano i test di Peter — contiene esattamente:

      {%- if enable_thinking is defined and enable_thinking is false %}
          {{- '<think>\n\n</think>\n\n' }}
      {%- endif %}

  Riga vuota fra i due tag, come sospettato. `think_suppression_prefix()` ora
  restituisce quella sequenza, e i byte esatti sono documentati sul posto con la
  citazione del template, così non si possono riperdere in un riordino.
  `main.rs:2975` chiama la funzione invece di ripetere il letterale, quindi la
  ricostruzione della history è rimasta allineata da sola.

  **Verifica before/after sulla stessa macchina, `temperature: 0`, stesso
  prompt dello smoke test** (`Count: one two three`, `num_predict 400`) — il
  before è stato ricompilato col prefisso vecchio proprio per non avere un
  controllo che passa in entrambi i casi:

  | prefisso | contenuto visibile | `</think>` al client |
  |---|---|---|
  | `<think>\n</think>\n\n` (vecchio) | 882 caratteri di ragionamento, poi il tag, poi la risposta | **sì** |
  | `<think>\n\n</think>\n\n` (nuovo) | `Count: one two three` | no |

  Il difetto era peggio di come l'avevamo scritto: non era «un tag che sfugge»,
  era che **la soppressione del thinking non funzionava affatto**. Con un solo
  newline il modello ragionava per intero nel canale visibile e chiudeva col tag;
  chi chiedeva `think: false` pagava i token del ragionamento e se lo vedeva in
  faccia. Verificato su tutti e tre i percorsi: NDJSON in streaming, non
  streaming, e motore sequenziale (`--batch-size 0`).

  **Resta aperto il punto 2**, la guardia sul tag orfano. Va nel `process_piece`
  condiviso di H2-D e non prima: metterla adesso significherebbe scriverla nello
  scheduler e lasciare scoperti gli altri due loop di decode — esattamente la
  duplicazione che H2-D esiste per togliere.

- [x] **H2-K · `run` e `serve` non devono partire con KV cache diverse** *(P1)*
  Emerso dai test di Peter sull'issue #140, non da una revisione del codice.
  `Commands::Run` aveva `cache_type_k`/`cache_type_v` a `f16`/`f16`,
  `Commands::Serve` a `q8_0`/`q4_0` (`main.rs:195-201` contro `main.rs:315-329`).
  Lo stesso modello dava quindi qualità di output diversa a seconda del comando
  che l'aveva avviato, in silenzio: niente nell'output lo diceva, e la V-cache a
  4 bit è aggressiva su Qwen3 al punto che un tester esterno ha visto
  degradazione con esattamente quella configurazione. Il commento nel codice
  giustificava la divergenza con «kept as-is to avoid changing default VRAM usage
  for existing serve users» — un ragionamento che protegge il consumo di memoria
  e sacrifica la correttezza dell'output, che è l'ordine sbagliato.
  Chiuso portando `serve` a `f16`/`f16`. Quantizzare la KV cache resta un trade
  supportato e utile a contesto lungo: deve essere una scelta dell'operatore, non
  un effetto collaterale del nome del comando. Tre test in
  `cli_default_parity_tests` bloccano la regressione, incluso uno che verifica
  che i default di `ctx_size` coincidano — stessa classe di divergenza, invisibile
  a chi ci sbatte contro.
  **Conseguenza operativa**: chiunque usi `eullm serve` con la configurazione di
  default vedrà ora più consumo di VRAM per la KV cache (circa 2× su K, 4× su V)
  e output migliore. Chi ha bisogno del comportamento vecchio lo ottiene con
  `--cache-type-k q8_0 --cache-type-v q4_0`, e ora sa di averlo chiesto.

- [x] **H2-L · Flash attention + key cache a 4 bit produce output incoerente** *(P1)*
  Partito dai report dell'issue #140 come «q4_0 degrada la qualità», che era la
  spiegazione sbagliata. Bisezionato in locale su CPU x86 con qwen3-0.6b a
  `temperature: 0`, una variabile alla volta:

  | configurazione | output |
  |---|---|
  | `f16`/`f16` | corretto |
  | `q8_0`/`q4_0` | corretto |
  | `f16`/`q4_0` | corretto |
  | **`q4_0`/`q4_0`** | **insalata di parole** |
  | **`q4_0`/`f16`** | **insalata di parole** |
  | `q4_0`/`q4_0` con `--no-flash-attn` | **corretto** |
  | `q4_0`/`q4_0` con `--batch-size 0` | insalata di parole |

  Quindi: non è la quantizzazione dei *valori* (f16/q4_0 va bene), non è lo
  scheduler (si riproduce identico in sequenziale), non è un backend (i report
  esterni lo mostrano su Metal e su CPU ARM). È **flash attention che non gestisce
  una key cache a 4 bit** e produce testo senza senso invece di rifiutare. Con
  `--no-flash-attn` la stessa configurazione risponde correttamente, incluse
  domande banali dove la differenza è inequivocabile («The capital of France is
  **Paris**» contro caratteri arabi casuali).

  **Perché è arrivato fin qui.** Il suggerimento `--cache-type-k q4_0
  --cache-type-v q4_0` nel messaggio d'errore di VRAM insufficiente
  (`scheduler.rs:387`) è stato introdotto in `6ea592b` del 10 luglio — **lo stesso
  commit che ha reso flash attention attiva di default**. La combinazione
  consigliata non ha quindi mai funzionato con la configurazione predefinita: è
  nata rotta. Non è un caso di «testato e poi regredito»: non è mai stata provata
  con FA attiva, perché FA attiva è arrivata nello stesso momento. E nessun test
  copriva la *qualità* dell'output sotto impostazioni KV non di default — lo smoke
  test verifica la correttezza solo ai default.

  Chiuso con una correzione automatica, non con un avviso:
  `correct_kv_cache_for_flash_attn` alza le chiavi a `q8_0` quando FA è attiva,
  spiegando a video cosa è successo e come ottenere davvero i 4 bit
  (`--no-flash-attn`). Un avviso avrebbe lasciato il processo a generare
  spazzatura, e chi ne ha più bisogno è proprio chi gira headless con l'output che
  finisce in una pipeline. Alzare K a q8_0 conserva flash attention e quindi il
  throughput, mentre disattivare FA di nascosto dimezzerebbe la velocità in
  silenzio — sorpresa peggiore di qualche centinaio di MB di VRAM. Stesso schema di
  `correct_kv_cache_for_model` per Gemma 4. Quattro test, incluso quello di
  idempotenza. Suggerimento e README allineati a `q8_0`/`q4_0`.

  **Lezione di metodo, la stessa del falso verde su H2-A:** un default e un
  suggerimento introdotti nello stesso commit non si verificano a vicenda. Nessun
  test guardava la qualità dell'output fuori dai default, quindi la combinazione
  che raccomandavamo attivamente non è mai stata eseguita da nessuno prima di un
  tester esterno, quindici giorni dopo.

- [x] **H2-M · Il default dei thread conta le CPU logiche, non i core** *(P1)*
  Emerso da una domanda sul report del MacBook Pro: quella macchina genera a
  **0,8 tok/s**, contro 11,7 del mac mini 2018 (stessa generazione di CPU, stesso
  binario) e 26,5 di un Raspberry Pi 5. Avevo liquidato i suoi timeout come
  «aritmetici, è solo lenta»: sbagliato. Un i9-8950HK 34 volte più lento di un
  Pi 5 non è una macchina lenta, è una macchina in una condizione anomala — e
  parte di quella condizione la creiamo noi.

  `threads.unwrap_or_else(available_parallelism)` chiedeva **tutte le CPU
  logiche**. `llama-cli` usa `common_cpu_get_num_math` →
  `common_cpu_get_num_physical_cores`, che su macOS legge
  `hw.perflevel0.physicalcpu`, cioè i core fisici *performance*. Sull'i9-8950HK
  fa 12 contro 6; sull'M1 fa 8 (4P+4E) contro 4.

  Misurato qui su una VM a 4 core senza SMT, dove la sovrasottoscrizione è
  l'unica variabile in gioco:

  | `--threads` | throughput |
  |---|---|
  | 1 | 16,4 tok/s |
  | 2 | 25,4 tok/s |
  | **4** | **41,9 tok/s** |
  | 8 | 16,9 tok/s |
  | 12 | 14,6 tok/s |

  Chiedere più thread di quanti core li possano eseguire costa il **60%** del
  throughput. Su Apple Silicon il meccanismo è anche peggiore: ggml divide il
  grafo equamente fra i thread, quindi ogni passo aspetta il più lento, e quattro
  core performance più quattro efficiency vanno **più piano** di quattro
  performance da soli. Su un portatile termicamente limitato poi si somma: più
  thread AVX-heavy significa più potenza assorbita, quindi clock sostenuto più
  basso — che è esattamente il difetto noto del MacBook Pro 15" 2018.

  Chiuso con `inference::default_thread_count`, che replica l'euristica di
  llama.cpp: core fisici performance su macOS via `sysctlbyname`, coppie
  `(physical id, core id)` distinte da `/proc/cpuinfo` su Linux, e il conteggio
  logico come ultima risorsa quando la piattaforma non sa rispondere — mai zero.
  Cinque test, incluso il caso ARM in cui il kernel omette `core id` del tutto.

  **Onestà su cosa è verificato**: la misura sopra è su una macchina senza SMT,
  quindi prova che la sovrasottoscrizione fa danno, non che il *nostro* nuovo
  default sia migliore su hardware ibrido — lì i due valori coincidono e il
  cambiamento è un no-op. La verifica vera arriva dai box di Peter con
  `--threads 6` sull'i9 e `--threads 4` sull'M1. Allineare l'euristica a quella
  dell'implementazione di riferimento è comunque la scelta conservativa, ed è
  anche ciò che rende sensato un confronto like-for-like con `llama-cli`.
  Prossimo sospetto sul gap con llama-cli, una volta chiuso questo.

- [x] **H2-N · `serve` di default dà 512 token di contesto a richiesta** *(P1)*
  `--ctx-size` è il budget KV **totale** e viene diviso equamente fra gli slot
  (`scheduler.rs:685-686`), mentre `serve` ha `--batch-size 8` di default contro
  l'1 di `run`. Risultato: `eullm serve` senza argomenti dà a ogni richiesta
  **4096 / 8 = 512 token**, `eullm run` gliene dà 4096. È la stessa classe di
  divergenza `run`/`serve` di H2-K, applicata al contesto invece che alla KV
  cache, e per un modello che ragiona è mutilante: Qwen3 spende più di 512 token
  solo nel `<think>`.
  Misurato con la stessa domanda su tre configurazioni:

  | configurazione | token generati | ctx per slot |
  |---|---|---|
  | `serve` default (8 slot) | **477, tagliata** | 512 |
  | `--batch-size 4` | 978, tagliata | 1024 |
  | `--batch-size 1` (come `run`) | 1055, completa | 4096 |

  Spiega direttamente i numeri nei report di Peter: `eval_count` 1001, 1014,
  1024 con `--batch-size 4` non sono coincidenze, sono il soffitto dello slot.
  Chiuso **non** cambiando il default alla cieca — sarebbe l'errore di H2-L
  un'altra volta — ma rendendo la divisione impossibile da non vedere: warning
  all'avvio con i due numeri che la risolvono («raise --ctx-size to 16384 or
  lower --batch-size»), soglia `MIN_COMFORTABLE_SEQ_CTX` a 2048. **La scelta dei
  default resta aperta** e va fatta con una misura di VRAM, non a intuito: il
  motivo per cui il contesto è totale e non per-sequenza è che la versione
  per-sequenza moltiplicava e andava in OOM (vedi il commento a
  `scheduler.rs:681-685`).

- [x] **H2-O · `done_reason` diceva sempre `"stop"`, anche quando mentiva** *(P1)*
  `done_reason` e `finish_reason` erano la stringa letterale `"stop"` in **undici
  punti** di `routes.rs`, e `StreamEvent::Done` non trasportava affatto il
  motivo: l'informazione non usciva nemmeno dallo scheduler. Una risposta tagliata
  perché lo slot ha esaurito il contesto era quindi **indistinguibile** da una che
  il modello ha deciso di terminare. È il difetto che ha reso invisibile H2-N per
  mesi, e che ha fatto leggere a un tester esterno delle troncature come «il
  modello si comporta male».
  Peggio: due dei siti erano il ramo `Err(_)` di un decode fallito. Un errore di
  decodifica a metà generazione veniva riportato al client come completamento
  regolare.
  Chiuso con `StopReason { Stop, Length }` propagato dallo scheduler *e* dal
  motore sequenziale fino a `done_reason`/`finish_reason`, con i due siti di
  errore convertiti in `StreamEvent::Error` (che entrambi i percorsi di streaming
  già sapevano rendere). Il default in `collect_stream` è `Length`, non `Stop`:
  se lo stream finisce senza un evento `Done` la risposta è incompleta, e l'ipotesi
  onesta è quella. La CLI interattiva stampa `truncated — out of context`.
  Verificato end-to-end: `num_predict: 8` → `done_reason='length'`, risposta che
  finisce da sola → `'stop'`, e il default di `serve` → `'length'` a 477 token.
  Due test unitari più due controlli nello smoke test.

- [x] **H2-P · `--daemon` dichiarava avviato un processo già morto** *(P1)*
  `daemonize` stampava «eullm daemon started (PID N)» nell'istante in cui
  `spawn()` ritornava, senza controllare nulla (`main.rs:3050-3062`). Se il figlio
  moriva subito — porta occupata è il caso comune — l'operatore riceveva comunque
  un messaggio di successo, un pidfile che punta a un processo inesistente e
  **exit code 0**. Lo script di chi testa prova allora a uccidere un PID che non
  c'è più, mentre il server *precedente* continua a rispondere: le richieste
  successive finiscono su un server avviato con flag diversi, senza che niente lo
  segnali.
  Non è teoria: è successo nei report dell'issue #140 su cinque macchine, ed è il
  motivo per cui non sappiamo con certezza quale server abbia risposto a quali
  query nei blocchi con `--cache-type-k q4_0`. Ha invalidato in silenzio parte di
  una raccolta dati fatta da qualcun altro con il suo tempo.
  Chiuso: dopo lo spawn si attende `DAEMON_STARTUP_GRACE` (1200 ms) controllando
  `try_wait()`. Se il figlio è già uscito si stampano il suo codice e le ultime
  dieci righe del suo log — che è dove sono finite le sue diagnostiche — non si
  scrive alcun pidfile (uno stantio è peggio di nessuno, perché uno script di stop
  ci crede) e si esce con 1. Verificato: avvio pulito → PID vivo ed exit 0; stessa
  porta due volte → errore del figlio riportato, nessun pidfile, exit 1.

- [x] **H2-Q · Il registro di audit si corrompeva sotto concorrenza** *(P0)*
  Trovato dallo smoke test al giro finale di verifica, non da una revisione:
  `audit records written — unreadable: Extra data: line 1 column 204`.
  `persist` usava `writeln!(file, "{json}")`, e `writeln!` passa da `write_fmt`,
  che emette **due syscall** — una per il valore formattato e una per il newline.
  Con `O_APPEND` ogni singola write è atomica, ma due scrittori concorrenti si
  intrecciano *fra* le due: il risultato è `{a}{b}\n\n`, cioè una riga con due
  record dentro e un file JSONL che non è più JSONL.
  Gravità: è il registro che esiste per essere una prova difendibile ai fini
  dell'AI Act, e `read_all` su un file così o fallisce o scarta record in
  silenzio. Il difetto si manifesta solo sotto carico concorrente, cioè
  esattamente in produzione e mai in un test manuale.
  Chiuso costruendo la riga completa e facendo **una sola** `write_all`. Test di
  regressione con 8 thread × 40 record e lunghezze variabili (così un intreccio
  non può essere mascherato da record tutti uguali): contro il codice vecchio
  fallisce con **275 righe su 320 — 45 record persi**; con il fix passa, e due
  esecuzioni consecutive dello smoke test danno 33 PASS / 0 FAIL.
  Da notare per il metodo: questo non lo avremmo mai trovato leggendo il codice
  o testando a mano. L'ha trovato un harness che manda 8 richieste insieme.

- [x] **H2-R · Con logit NaN generavamo spazzatura e la chiamavamo risposta** *(P1)*
  Il log di `--rust-debug` dal mac mini 2018 dell'issue #140 ha finalmente detto
  cos'erano gli `@@@`: **tutti e 151.936 i logit sono NaN**, in 5.020 casi su
  5.021, dal primo token della prima richiesta. Con una distribuzione interamente
  NaN i confronti del sampler degenerano sempre sullo stesso token — quello è
  `@`. Su quella macchina non è mai uscita una risposta sensata: 4 risposte di
  soli `@`, 2 errori `Decode Error -3` (che è `GGML_STATUS_FAILED` da
  `graph_compute`), zero altro.
  Il difetto **nostro** non è il NaN: è che lo sapevamo e consegnavamo comunque
  1014 token di `@` con `done_reason: "stop"`. Il rilevatore aggiunto in 0.6.33
  scansiona tutti i logit e per questo sta dietro `--rust-debug`; senza quel flag
  — cioè per tutti — non c'era nessun controllo.
  Chiuso con una guardia **O(1) sempre attiva**: dopo il sampling si guarda il
  logit del token appena scelto. Se è NaN o Inf, la distribuzione da cui è stato
  scelto era priva di senso, e la richiesta fallisce con un errore esplicito
  invece di generare spazzatura. Un valore, non 150k: costo nullo per token.
  Inserita in tutti e cinque i punti di sampling — i due dello scheduler
  (prefill e decode loop) e i tre del motore sequenziale, altrimenti
  `--batch-size 0` restava scoperto. La KV cache della sequenza viene azzerata,
  non offerta al riuso di prefisso: qualunque cosa abbia prodotto NaN è dentro.
  **Non è una riparazione, ed è importante non spacciarla per tale.** Il NaN
  resta. Serve a non mentire, esattamente come `done_reason`, e a impedire che
  una risposta corrotta entri in una pipeline dove nessuno la guarda — che è il
  caso che conta se il fenomeno è intermittente invece che totale come su quella
  macchina. La riparazione vera dipende da quale dei tre rami è, e nessuno dei
  tre è ancora escluso: hardware di quella macchina, kernel di llama.cpp
  (AVX2 o flash attention su quella CPU), o come guidiamo noi llama.cpp. Tre run
  da un flag ciascuno lo dicono: `--no-flash-attn`, `--batch-size 0`,
  e `llama-cli` sullo stesso GGUF.
  Nota sulla sicurezza dell'implementazione: `get_logits_ith` ha un `assert!`
  interno sull'indice, e ora gira a ogni token invece che sotto un flag di debug.
  L'invariante che lo rende sicuro è documentata sul posto — l'indice è quello
  registrato da `decode_batch.add(..., true)`, ed è lo stesso che
  `LlamaSampler::sample` ha appena usato una riga sopra. Verificato con 8
  sequenze concorrenti nello smoke test, che è il percorso multi-sequenza dove
  un indice sbagliato si vedrebbe.

- [x] **H2-S · L'unico artefatto della matrice compilato per un'architettura che
  il runner non ha** *(P1)*
  **Chiusa il 29 luglio 2026.** Le due macchine di Peter funzionano: la tabella
  della issue #140 riporta `eullm-macos-x64` come *Tested (community)* con il
  mac mini 2018 i7-8700B a 54 tok/s e il MacBook Pro 15" i9-8950HK a 41 tok/s.
  Il merito è quasi certamente di **H2-T** (il binario caricava 29 layer su GPU
  via Metal mentre dichiarava CPU), chiusa in 0.6.39 una release dopo il cambio
  di runner. Le due modifiche non sono separabili dai dati raccolti, perché
  nessuno ha provato la 0.6.38 da sola, e isolarle costerebbe a un tester il
  download di una release vecchia per un'informazione che non cambia nulla:
  `macos-15-intel` resta comunque, perché è quello che usa llama.cpp a monte e
  perché toglie una classe di confusione host/target, non perché sia
  dimostrato che servisse.
  Prima di questo punto ci sono tre giorni di ipotesi sul NaN degli Intel Mac
  senza aver mai guardato né l'issue tracker di llama.cpp né come llama.cpp
  stesso pubblica il proprio binario macOS x64. Errore di metodo, non di stile:
  due delle quattro ipotesi si chiudevano leggendo il nostro `build.rs`, una
  terza con una ricerca. Registrato qui perché si ripete facilmente.

  Cosa dice il confronto con monte:
  - llama.cpp costruisce il proprio `macos-x64` su un runner **Intel nativo**
    (`macos-15-intel`), con `-DGGML_METAL=OFF`. Non lo cross-compila.
  - noi lo costruivamo su `macos-15`, che è Apple Silicon. Era **l'unico
    artefatto della matrice** compilato per un'architettura che il runner non
    possiede, e l'unico che non ha un percorso di cross-compilazione gestito
    esplicitamente in `build.rs` (per `aarch64-unknown-linux-gnu` c'è: forza
    `GGML_NATIVE=OFF` e fissa `GGML_CPU_ARM_ARCH`; per x86_64-su-arm64-macOS
    niente). Nota contro me stesso: il binario linux-arm64 è anch'esso
    cross-compilato e sul Pi 5 di Peter va a 26,5 tok/s — quindi
    «cross-compilare rompe» non è una legge, è un sospetto su *questo* percorso.
  - `issue #9873` di llama.cpp è la stessa classe di sintomo: su macOS 15 Intel
    un binario **compilato in casa** produce token spazzatura (`iorimondeaphans1
    联…`) mentre il binario **ufficiale** risponde bene, stesso modello. Aperta,
    `bug-unconfirmed`. In quel caso la differenza plausibile è Metal (l'utente
    aveva Metal ON su una Iris Plus, l'ufficiale OFF), quindi **non spiega** il
    risultato di Peter su 0.6.36 — ma dimostra che su macOS 15 Intel
    «l'ambiente di build del binario, non la macchina» è un modo di guasto reale.

  Cambiato: `x86_64-apple-darwin` passa da `macos-15` a `macos-15-intel`
  (GA fino ad agosto 2027, ultima immagine x86_64 che GitHub offrirà; `macos-13`
  è ritirata). Una riga, e va in 0.6.38 da sola. **Non è dichiarata come la
  soluzione del NaN**: è la rimozione dell'ultima variabile di build non
  esaminata, e serve la release per sapere se cambia qualcosa sulle due
  macchine di Peter.

  Escluse leggendo il nostro codice, non ipotizzando:
  - **Accelerate / BLAS**: `build.rs` mette `GGML_BLAS=OFF` su *tutti* i target
    Apple. Il prefill non passa da `cblas_sgemm`. Non è quello.
  - **Metal su GPU non-Apple**: già rimosso da questo target il 2026-07-22
    (`c891fc8`, v0.6.30), cioè **prima** della 0.6.36 che Peter ha provato. È
    invece la spiegazione più probabile delle sue primissime segnalazioni: su
    Intel Mac è un guasto documentato a monte, sia su AMD discrete
    (`issue #19563`, Radeon Pro 5300M — stessa famiglia del MacBook Pro 2018)
    sia via Vulkan/MoltenVK (`issue #20104`, «produce gibberish»).
  - **Un bug noto di llama.cpp su Coffee Lake CPU-only**: non esiste. Nessuna
    segnalazione a monte di logit tutti NaN su un build CPU-only x86. L'assenza
    conta: rende meno probabile «llama.cpp è rotto su quella CPU» e più
    probabile qualcosa di specifico del nostro binario o di quelle due macchine.

- [x] **H2-T · Il binario macOS x64 non è mai stato CPU-only, e lo dicevamo a
  tutti** *(P0)*
  La voce più costosa di tutto il documento, perché per un mese l'abbiamo
  affermata al contrario — a noi stessi, nel codice, nella tabella della
  release e nell'issue #140 a un tester esterno.

  Il log `--rust-debug` della 0.6.38 dal MacBook Pro 2018 mostra, a distanza di
  quindici righe:

      ║  WARNING: GPU requested but this binary has no GPU support  ║
      ║  All inference will run on CPU (very slow for large prompts) ║
      WARN eullm::inference: No GPU backend compiled (gpu_layers=-1).
      ggml_metal_device_init: GPU name:   MTL0 (AMD Radeon Pro 560X)
      load_tensors: offloaded 29/29 layers to GPU
      llama_kv_cache: MTL0_Private KV buffer size = 448.00 MiB

  E sul mac mini 2018, identico, su `Intel(R) UHD Graphics 630`.

  **Due difetti indipendenti, entrambi nostri, che si sono sommati:**
  1. `cfg!(feature = "metal")` non ha mai controllato il backend. Il CMake di
     ggml mette `GGML_METAL=ON` di default su *ogni* target Apple, e il
     `build.rs` di `llama-cpp-sys-2` lo spegneva solo per watchOS. Togliere la
     feature dal target x64 in v0.6.30 non ha quindi tolto niente: il backend
     Metal continuava a essere compilato e il dispositivo enumerato.
  2. `check_gpu_support` stampava l'avviso e **non cambiava niente**. Con
     `gpu_layers = -1` (il default) il ramo `else` passava
     `with_n_gpu_layers(1000)` comunque. Su Linux e Windows senza feature GPU
     è innocuo, perché non c'è nessun backend su cui scaricare. Su macOS no.

  Quindi da v0.6.30 in poi le due macchine Intel di Peter hanno girato tutto il
  modello su Metal, su una UHD 630 e su una Radeon Pro 560X. È esattamente la
  configurazione che a monte è documentata come produttrice di risultati
  sbagliati (`ggml-org/llama.cpp#19563` su Radeon Pro, `#4004` su Intel Mac in
  generale), e llama.cpp costruisce la propria release macOS x64 con
  `-DGGML_METAL=OFF` proprio per questo. Il log lo conferma anche a livello di
  capacità del dispositivo: `simdgroup matrix mul. = false` su entrambe.

  Chiuso su tutti e due i fronti, perché uno solo non basta:
  - **build**: `build.rs` ora rispetta la feature `metal` su tutti i target
    Apple, non solo watchOS. Senza feature il backend non viene proprio
    compilato. La catena `eullm/metal → llama-cpp-2/metal →
    llama-cpp-sys-2/metal` inoltra correttamente, quindi il target arm64
    (che la feature ce l'ha) non cambia.
  - **runtime**: `check_gpu_support` restituisce il numero di layer che si
    possono davvero chiedere — 0 se nessun backend è compilato — ed è
    `#[must_use]`. I due punti di caricamento (motore sequenziale e
    scheduler) usano quel valore invece di `config.gpu_layers`, e la stessa
    regola vale per `use_gpu` del percorso multimodale. Vale su ogni
    piattaforma e indipendentemente da come il binario è stato costruito, che
    è ciò che serve: il difetto 1 era invisibile proprio perché ci fidavamo
    di una feature al posto di un controllo.

  **Nota di metodo, la parte scomoda — e non è quella che sembra.** La
  diagnosi non è arrivata tardi: è arrivata il **22 luglio**, due giorni dopo
  il primo report di Peter, ed era giusta. Quel giorno abbiamo rilasciato
  quella che credevamo fosse la correzione (0.6.31, commit `c891fc8`): **una
  riga cancellata**, `features: metal`, dalla matrice di build. Peter ha
  provato e ha risposto «no changes from 0.6.30».
  **Lì sta l'errore.** Un risultato negativo dopo una correzione ha due
  spiegazioni — l'ipotesi è sbagliata, oppure la correzione non ha fatto
  effetto — e ne abbiamo considerata una sola. Abbiamo scartato l'ipotesi
  giusta e siamo andati a cercare altrove per quattro giorni: baseline AVX2,
  default dei thread, tipi di KV cache, guardia NaN, cross-compilazione.
  Quello che avrebbe chiuso tutto è un `grep ggml_metal` sul binario
  pubblicato. Il 22 luglio ho scritto «eullm-macos-x64 è costruito senza
  Metal, quindi non tocca mai la GPU» senza mai averlo verificato su un
  binario — inclusa la caccia alla cross-compilazione di H2-S, che resta
  corretta come igiene ma non era questo. Non avevo mai letto un log di avvio
  di quel binario su una macchina Apple: se l'avessi fatto, `ggml_metal_init`
  era lì dalla prima riga. La lezione non è «Metal è insidioso», è che una
  feature di compilazione non è una prova di cosa fa il binario, e il log di
  avvio del binario vero sì.

  Resta da verificare sul campo che con Metal spento davvero le due macchine
  producano risposte valide. Finché non arriva quel dato, i NaN sono
  **spiegati in modo plausibile, non dimostrati**.

- [x] **H2-U · Il filtro Harmony non copriva la forma che il modello emette
  davvero** *(P2)*
  Trovata eseguendo il percorso multimodale per la prima volta, non leggendo il
  codice. Gemma 4 12B, richiesta reale con un'immagine, ha risposto:

      <|channel>thought\n<channel|>A small, metallic electronic device...

  C'è un **newline** fra `thought` e `<channel|>`. Il pattern combinato in
  `DEFAULT_HARMONY_FILTERS` era `"<|channel>thought<channel|>"`, senza newline,
  e siccome il filtraggio è per sottostringa letterale non agganciava niente.
  L'impalcatura arrivava all'utente.

  **Non è una regressione di H2-D**, ed è importante non raccontarla come tale:
  prima della 0.6.40 su quel percorso i filtri **non venivano applicati
  affatto**, quindi quel testo usciva comunque, e peggio. H2-D ha fatto arrivare
  i filtri fin lì; questa voce è il pattern che, arrivato, non bastava.

  Aggiunte le tre varianti con newline. Due test con i byte esatti osservati:
  uno verifica che il preambolo sparisca e la risposta resti, l'altro che un
  blocco **con contenuto** continui a passare — la UI lo rende come sezione di
  ragionamento, e togliere i delimitatori nudi lascerebbe il ragionamento come
  testo senza marcatore.

  Seconda cosa emersa dallo stesso test: un `.webp` falliva con
  `Media #0 failed to decode: NullResult`, che non dice niente. Il backend
  multimodale decodifica jpg, png, bmp e gif; ora l'errore lo scrive, invece di
  lasciare che la risposta stia nel sorgente.

  Nota di metodo: `cargo check --features multimodal` dimostra che quel codice
  compila e nient'altro. Il difetto era invisibile a 191 test e visibile alla
  prima richiesta vera.

- [x] **H2-V · Un `manifest.json` danneggiato nascondeva tutti i modelli** *(P1)*
  Segnalato dal campo, non da una revisione: `eullm list` rispondeva
  `Error listing models: expected ',' or '}' at line 24 column 3` e basta.
  Nessun elenco, e nessuna indicazione di quale directory fosse il problema.

  Due difetti che si alimentano a vicenda, ed è importante vederli entrambi
  perché correggerne uno solo lascia il sistema fragile:

  1. **Scrittura non atomica** (`store.rs`): `fs::write` tronca il file e poi
     scrive. Un processo che muore in mezzo — crash, `kill`, disco pieno —
     lascia un percorso valido con dentro contenuto invalido. È così che il
     manifest si rompe.
  2. **Lettura intollerante**: `list()` faceva `serde_json::from_str(&data)?`,
     e quel `?` propaga. **Un** manifest rotto rendeva invisibili **tutti** gli
     altri modelli. Le due funzioni sorelle nello stesso file
     (`store.rs:186` e `:221`) erano già tolleranti: la sola severa era quella
     che l'utente lancia davvero.

  Chiuso su entrambi i fronti. La scrittura passa da `write_atomically`, che
  scrive su un file temporaneo **nella stessa directory** — un rename fra
  filesystem non è atomico e ricadrebbe in una copia, cioè proprio il modo di
  guasto da evitare — e poi rinomina. `std::fs::rename` sostituisce la
  destinazione esistente sia su Unix sia su Windows, quindi il lettore vede o
  il vecchio manifest o il nuovo, mai un prefisso di uno dei due. La lettura
  salta il manifest illeggibile con un warning che **nomina il file** e dice
  cosa fare.

  Verificato riproducendo il sintomo su un archivio con un manifest troncato:
  il binario pubblicato stampa solo l'errore del parser, quello corretto elenca
  il modello sano e nomina quello rotto. Quattro test, inclusi i due che
  coprono l'atomicità.

  Nota sul fixture: la prima versione del test passava per il motivo sbagliato,
  perché il manifest "sano" che avevo scritto era incompleto e veniva scartato
  anche lui — l'elenco risultava vuoto e l'asserzione lo prendeva per un
  successo. Un test che non distingue i due casi non verifica niente.

- [x] **H2-W · Niente diceva quale archivio di modelli era in uso** *(P1)*
  Trovata inciampandoci: `eullm list` mostrava `gemma-4-e4b ready` mentre l'API
  rispondeva 404 per lo stesso nome. **Entrambe avevano ragione**, perché i due
  processi leggevano directory diverse — una con `EULLM_MODELS_DIR` impostata,
  l'altra senza. Nessuno dei due lo diceva, quindi la diagnosi è costata tre
  scambi e un manifest ricostruito a mano per un modello che stava altrove.

  L'incoerenza era già visibile nel codice: all'avvio il server stampa la
  destinazione dell'**audit**, l'allowlist degli IP, le origini permesse. Non
  la directory dei modelli, che è quella su cui risolve ogni richiesta.

  Chiuso stampandola dove serve. `list` intesta l'elenco con
  `Models in <root> [EULLM_MODELS_DIR|default]`, e lo dice anche quando
  l'elenco è vuoto — che è il caso in cui serve di più. Il server la registra
  all'avvio accanto alle altre righe di configurazione.

  **Secondo difetto trovato nello stesso giro**: `list` riportava `status`
  copiandolo dal manifest, che è una stringa scritta al momento del pull. Un
  modello il cui GGUF non c'è più, o non è mai arrivato, restava `ready` per
  sempre. Ora la colonna guarda il disco: `ready (file missing)` quando il file
  che il manifest nomina non esiste. È lo stesso vizio di H2-T e del banner GPU
  — riportare ciò che è scritto da qualche parte invece di ciò che è vero — e
  fa la terza volta in due giorni.

- [x] **H2-X · Il selettore interattivo cercava i modelli con la chiave
  sbagliata, nella directory sbagliata** *(P1)*
  Trovata guardando la schermata di `eullm` lanciato senza argomenti mentre
  cercavamo tutt'altro: la sezione `LOCAL` mostrava **una riga sola**, e i
  modelli che sapevamo essere su disco non c'erano.

  Tre difetti sovrapposti in quaranta righe, e tutti e tre della stessa
  famiglia di H2-W — chiedere a una fonte che non è il disco.

  1. **La chiave.** `list_local_ggufs` chiamava `store.gguf_path(&m.name)`.
     `name` è il titolo leggibile («DeepSeek R1 Distill (Qwen-14B)»),
     `gguf_path` lo usa come **nome di directory**. Nessuna directory si
     chiama così, quindi ogni modello del catalogo presente su disco
     spariva da `LOCAL`. Sopravvivevano solo quelli il cui titolo coincide
     per caso con l'id: da lì la riga singola. Ora la chiave è `m.id`, con
     `name` come ripiego per i manifest antecedenti al campo `id`.
  2. **La directory.** Il ramo che raccoglie i `.gguf` sciolti nella radice
     dell'archivio aveva `$HOME/.eullm/models` scritto a mano, quindi
     ignorava `EULLM_MODELS_DIR` e guardava in una cartella che il resto del
     processo non stava usando. Ora chiede la radice all'archivio
     (`root_with_source`). È esattamente l'incoerenza di H2-W, in un punto
     che quella correzione non aveva toccato.
  3. **Il tag `[local]`.** La marcatura delle voci di catalogo già scaricate
     usava `store.exists`, che verifica solo la presenza di `manifest.json`.
     Il manifest viene scritto **prima** che il download finisca e resta lì
     se il file dei pesi viene cancellato: il tag prometteva pronti dei
     modelli che non partono. Ora usa `is_present`, che guarda il GGUF.

  Nota di metodo: il selettore è la prima cosa che vede chi installa EuLLM e
  non ha alcun test. Non è stato scoperto da un'analisi ma perché l'utente ha
  incollato la schermata mentre indagavamo su un 404. Coperto con tre test
  sull'archivio (risoluzione per id e non per titolo, radice con la sua
  provenienza, manifest senza il proprio GGUF).

- [x] **H2-Y · La chat non riconosceva il modello caricato** *(P0)*
  Segnalata dall'uso, non da una revisione: si lancia `eullm`, si sceglie un
  modello dal picker, si carica, si apre la chat e **ogni messaggio** risponde
  `No model loaded`. Il pannello di stato diceva `(none loaded)` mentre il
  modello era in memoria e rispondeva benissimo alle chiamate dirette all'API.

  `/api/tags` segnala il modello nello slot mettendolo per primo **con il
  digest vuoto**. Poi, se quel modello è nel catalogo, riscrive quella voce con
  i metadati completi — digest vero compreso. La UI riconosceva il modello
  caricato **proprio dal digest vuoto**, quindi la riscrittura cancellava
  l'unico segnale disponibile: la voce diventava indistinguibile da un modello
  mai scaricato e finiva fra quelle disabilitate.

  Perché è sopravvissuta tanto: un modello avviato da percorso
  (`eullm run ./model.gguf`) non passa mai da quel ramo, il digest resta vuoto
  e tutto funziona. Si rompe **solo** scegliendo dal catalogo, che è il modo in
  cui il prodotto si presenta a chi lo installa. Chiuso mettendo `loaded` nella
  risposta invece di farlo dedurre.

  **Secondo difetto trovato nello stesso giro**, indipendente: con `eullm serve`
  la tendina non offriva *niente* di selezionabile, perché ogni voce di catalogo
  era disabilitata come "non ancora scaricata" che tu l'avessi o no. Il server
  sapeva già fare lo swap alla prima richiesta; mancava solo il modo di dirglielo
  dalla UI. Ora `/api/tags` riporta anche `downloaded`, chiedendolo all'archivio,
  e l'elenco separa ciò che è su questa macchina da ciò che sarebbe un download.

  **Terzo, sullo stesso percorso**: un vocale WhatsApp è Ogg/Opus, che miniaudio
  non legge, quindi arrivava al motore come byte non decodificabili. Le immagini
  fuori formato venivano già riconvertite nel browser; ora lo è anche l'audio,
  in WAV mono a 16 kHz. Verificato sul campo: trascrizione corretta di un
  parlato in italiano, 4,25 s su CUDA e 143 s su CPU, stessa qualità.

  Nota di metodo: la chat UI non ha un solo test, ed è la prima cosa che vede
  chi installa EuLLM. Tre difetti in un pomeriggio, tutti trovati usandola.

- [x] **H2-Z · Un `pull` che non può creare la sua directory incolpava
  HuggingFace** *(P2)*
  `eullm pull gemma-4-e4b` rispondeva `Download failed: File exists (os error
  17)` e subito sotto «This may be because the model hasn't been published
  yet». **Sbagliate tutte e due le metà**: l'errore veniva da `create_dir_all`
  sulla directory del modello, e il modello sul server non c'entrava niente.

  `create_dir_all` restituisce `AlreadyExists` in due casi che si leggono
  identici — il percorso occupato da un file normale, oppure un symlink il cui
  bersaglio non esiste più — e nessuno dei due è indovinabile da «File exists».
  Ora la directory viene creata **prima** di qualsiasi richiesta HTTP, il
  messaggio nomina il percorso e dice cosa c'è sopra, e il suggerimento sul
  server non viene stampato per un problema locale.

- [x] **H2-AA · Il banner riportava metà della memoria KV su Qwen3** *(P2)*
  Trovata nel log di avvio di un utente esterno (issue #286), non da un
  controllo nostro, e visibile a due righe di distanza nello stesso schermo:

      llama_kv_cache: K (f16): 224.00 MiB, V (f16): 224.00 MiB   ← llama.cpp
        KV memory:     K=112 MiB, V=112 MiB                       ← noi

  Calcolavamo la dimensione della testa come `n_embd / n_head`. Quello è il
  **default** del formato GGUF, non una definizione: un modello può dichiarare
  `attention.key_length` e `value_length` per conto suo, e Qwen3 lo fa —
  1024 su 16 teste con lunghezza dichiarata 128, non i 64 che dà la divisione.
  Quindi **ogni modello Qwen3** veniva riportato a metà del suo costo reale.

  Ora leggiamo il valore dichiarato, con la divisione come ripiego per i
  modelli che non lo portano. La cache era sempre stata allocata giusta da
  llama.cpp: sbagliato era solo il numero che mostravamo. Contava lo stesso,
  perché quel numero serve a scegliere una finestra di contesto che ci stia, e
  sbagliava **per difetto** — la direzione che ti porta a chiederne una che non
  entra.

  Nota di metodo: è il terzo difetto di fila trovato leggendo l'output di
  avvio di qualcun altro, dopo i Mac Intel e il GPU backend. Continuare a
  chiedere il log completo invece delle sole risposte dell'API è la pratica
  che sta pagando di più.

- [x] **H2-AB · Il banner stampava meno per certi modelli, senza dirlo** *(P2)*
  Sullo stesso schermo, `gemma-4-e4b` su CPU: sotto `Context:` non compariva
  né la riga `KV memory:` né l'avviso sul contesto di addestramento. Non erano
  rotti, non venivano proprio prodotti — entrambi i numeri li calcolava lo
  **scheduler**, e un modello multimodale carica sempre in modalità
  sequenziale, come `--batch-size 0`.

  La parte che conta non è la riga mancante ma il silenzio: nessuna riga
  diceva "questi numeri non sono disponibili", quindi sembrava un modello che
  non aveva niente da riportare. È lo stesso vizio del `GPU layers: all` sui
  Mac Intel — un output che tace invece di dichiarare.

  Chiuso condividendo la stima: il percorso sequenziale la chiede al
  caricamento. Da notare che io stesso avevo proposto quelle due righe come
  "verifica gratuita" senza accorgermi che su quel modello non potevano
  comparire: la prova era stata scritta senza guardare quale ramo di codice
  la produceva.

- [x] **H2-AC · Gli elenchi dei modelli ignoravano il disco** *(P1)*
  Segnalata come issue #294: un modello scaricato da URL è utilizzabile dalla
  chat ma non compare in `/v1/models`, quindi non è selezionabile da un editor.

  Entrambi gli endpoint erano costruiti dal catalogo interno più il modello
  eventualmente caricato in quel momento, e **non guardavano mai l'archivio**.
  Un modello preso da un URL o da un repo HuggingFace era invisibile a
  entrambi se non era caricato proprio allora.

  Su `/v1/models` non è una questione estetica: **un plugin da editor offre i
  modelli che quell'endpoint nomina**, quindi un modello non nominato non è
  assente da una lista, è irraggiungibile, e l'utente non ha niente da
  digitare per aggirarlo. Ora entrambi elencano prima ciò che è su disco, poi
  il modello caricato, poi il catalogo, saltando i duplicati.

- [x] **H2-AD · Il proiettore multimodale era raggiungibile solo dal catalogo**
  *(P1)*
  Emersa dal confronto che ha fatto l'utente: «lo stesso prompt, immagine e
  modello funzionano con llama.cpp». Con llama.cpp il proiettore glielo passi
  con `--mmproj`. **Noi quel flag non ce l'avevamo.**

  `mmproj_path` cercava dentro l'archivio, per id del modello. Funzionava per
  i modelli scaricati dal nostro catalogo e per nient'altro: un GGUF preso a
  mano non poteva essere multimodale nemmeno con `mmproj-F16.gguf` nella stessa
  cartella — che è il layout di ogni repo vision su HuggingFace. Quindi «con
  llama.cpp funziona» non era una differenza di qualità, era un flag mancante.

  Chiuso su tre fronti: un `mmproj*.gguf` accanto ai pesi viene usato da solo,
  `--mmproj <path>` copre il caso in cui i due file stanno separati (su `run`
  **e** su `serve`, regola di parità), e il rifiuto ora nomina il modello e
  dice entrambe le strade invece di parlare di modalità interne del loader.

  Rischio dichiarato nel messaggio e nell'aiuto del flag, perché fallisce in
  silenzio: un proiettore appartiene ai pesi su cui è stato addestrato.
  Accoppiarlo a pesi diversi **non dà errore**, dà risposte convinte e
  sbagliate. Su `serve` si applica solo ai modelli che non ne hanno uno
  proprio, e ogni volta che si applica viene loggato.

  **Difetto gemello, stesso giro**: una build senza la feature `multimodal`
  scartava l'array `images` in silenzio e passava la domanda come testo. Il
  modello, interrogato su un'immagine mai ricevuta, rispondeva di non vederla
  — e sembrava un limite del modello. Ora `multimodal` è feature di default
  (tutti i binari pubblicati ce l'avevano dalla 0.6.42; solo le build da
  sorgente no) e una build che davvero non ce l'ha risponde 501 nominando il
  flag mancante.

### Verificato e **non** indotto da noi

Elencato perché l'assenza di un difetto va registrata quanto la presenza, e
perché ognuna di queste era un sospetto plausibile:

| sospetto | verdetto |
|---|---|
| Politica flash attention | **Corretta.** Passiamo AUTO (-1), non ENABLED, quindi llama.cpp fa la sua verifica di compatibilità (`inference/mod.rs:134-149`). Che poi AUTO non intercetti le chiavi a 4 bit è un limite a monte, non una nostra forzatura. |
| BOS aggiunto due volte | **No.** `AddBos::Always` è `add_special=true`, e `llama_tokenize` lo applica solo se il vocabolario dichiara `add_bos_token` — che Qwen3 mette a false. |
| Divisione del contesto = bug | **No, è il comportamento anche di `llama-server`.** Il difetto è il *default* (H2-N), non la semantica. |
| Overhead dello scheduler su singola richiesta | **Nessuno di misurabile**: 42,7 tok/s con `--batch-size 4` contro 39,6 sequenziale, stessa macchina. |
| Default di sampling diversi da llama.cpp | **Diversi ma deliberati**: seguiamo Ollama (temp 0.8, top_k 40, top_p 0.9, repeat_penalty 1.1) mentre llama.cpp usa top_p 0.95, `min_p 0.05`, repeat_penalty 1.0. L'unico che vale la pena riconsiderare è `min_p`: a 0.0 non filtriamo la coda della distribuzione, e su una macchina dai numeri marginali è meno protettivo. Da valutare, non un bug. |

---

## H3 — Igiene di processo e supply chain

**Gate di uscita:** la CI verifica ciò che la Definition of Done afferma; il mirror
garantisce ciò che dichiara; le dipendenze sono controllate per costruzione e non per
diligenza manuale.

- [x] **H3-A · La CI deve compilare le feature che può compilare a costo basso** *(P1)*
  *Riformulata dopo aver misurato i costi reali: la CI gira solo su `push: [main]`,
  in ~7 minuti, su runner standard di un repo pubblico — quindi gratuiti. Il
  vincolo non sono i minuti ma il wall-clock, e le feature GPU richiedono il
  toolkit CUDA che porterebbe quei 7 minuti a 25+ su ogni push. Ambito corretto:
  `cargo check --features multimodal` a ogni push (nessun toolkit, costo
  trascurabile), feature GPU su un job settimanale schedulato o lasciate al
  workflow di release.*
  La Definition of Done di `roadmap-engine-0.7-1.0.md` richiede che ogni voce «non rompa i
  feature flag CUDA, Metal, ROCm, Vulkan, multimodal», ma la CI esegue `cargo build` e
  `cargo test` solo con le feature di default (`.github/workflows/ci.yml:64, 67`): nessun
  percorso GPU e nessuna riga `#[cfg(feature = "multimodal")]` è mai compilata prima del
  push di un tag. È una Definition of Done non presidiata, in un repository che ha già
  imparato quanto costa scoprire un problema durante una release. Aggiungere un job
  `cargo check --features <x>` per ciascuna feature (economico: nessun link finale) e
  `cargo test --features multimodal` dove i test non richiedono GPU.

  **Chiusa dal modo peggiore: rompendo una release.** Il refactor di `StopReason`
  ha toccato `GenerateResult` e `StreamEvent::Done`, tipi usati anche da codice
  dietro `#[cfg(feature = "multimodal")]`. Quel codice è invisibile a
  `cargo build`, `cargo test` e `cargo clippy` senza la feature: tutto verde in
  locale, tutto verde sulla CI di main, e poi **tre job CUDA falliti al tag**,
  perché i build di release usano `--features "cuda,multimodal"`. Tre build
  lunghi e una release da rifare per un errore che `cargo check` trovava in un
  minuto.
  Aggiunto `cargo check --features multimodal` al job engine. `check` e non
  `build` perché il fallimento da prevenire è di tipi, non di link; e `cuda`
  resta fuori di proposito — richiede il toolkit CUDA, minuti di installazione a
  ogni push, e condivide lo stesso codice Rust che questo step già copre.
  Nota per chi ci tornerà: `cargo clippy --features multimodal` segnala oggi due
  lint preesistenti in `inference/mod.rs:867` e `:1578`, in codice mai passato
  sotto lint. Vanno sistemati prima di poter alzare anche clippy alla feature.

- [x] **H3-B · Il mirror non deve poter sovrascrivere i tag** *(P1)*
  **Chiusa il 30 luglio 2026.** Il push è spezzato in due: `--force` va sul
  solo `master`/`main`, i tag vanno senza. Un tag riscritto a monte viene
  quindi **rifiutato**, che è l'esito voluto — il mirror conserva l'oggetto a
  cui il nostro lockfile si riferisce — e la cosa è segnalata con un warning
  invece di far fallire il job, perché gli altri tag della stessa invocazione
  sono arrivati.

  Il commit pinnato del submodule è ancorato sotto `refs/eullm/pinned/<sha>`,
  letto dal gitlink di questo repository. Un force-push del branch non può
  orfanare un ref che non è quel branch, quindi la garbage collection non se
  lo può prendere. L'operazione è idempotente: il ref è indirizzato dallo SHA
  che nomina.

  Lo step di verifica non si limita più a confrontare le SHA del branch, che
  dice solo che il branch è arrivato e niente sull'unico oggetto la cui
  perdita conterebbe: ora controlla che l'ancora del commit pinnato esista sul
  mirror, e **fallisce** se non c'è.

  Su `llama-cpp-rs` c'è lo stesso trattamento dei tag ma nessuna ancora:
  è vendorizzato come sorgente sotto `engine/vendor`, non è un submodule,
  quindi non esiste un gitlink da ancorare.
  `mirror-sync.yml:9-11` dichiara che i tag sono immutabili e che ogni versione da cui si è
  dipeso resta pinnabile anche se l'upstream sparisce. Il comando che lo implementa è
  `git push --force … master --tags` (`:35-37, 76-78`), e `--force` si applica a tutti i ref
  della stessa invocazione, tag inclusi: un tag riscritto a monte viene sovrascritto anche
  sul mirror entro 24 ore. Inoltre nessun ref protegge il commit pinnato: se l'upstream fa
  rebase di `master`, il force-push lo orfana e la garbage collection può rimuoverlo —
  esattamente lo scenario contro cui il mirror esiste. Applicare `--force` solo a
  `master`/`main` e mai ai tag, e mantenere un ref `refs/eullm/pinned/<sha>` per ogni commit
  da cui si è mai dipeso. Estendere lo step di verifica: oltre al confronto delle SHA,
  controllare che il commit del submodule sia ancora raggiungibile sul mirror.

- [x] **H3-C · Controlli automatici su dipendenze e licenze** *(P1)*
  **Chiusa il 29 luglio 2026.** `deny.toml` alla radice del workspace e job
  `deps` in `ci.yml` (`cargo deny check advisories bans licenses sources`),
  fuori dal filtro `changes` perché una dipendenza diventa vulnerabile senza
  che nessuno tocchi il repository. L'allowlist delle licenze è generata dalle
  licenze realmente presenti, non da un template, ed è per costruzione: quello
  che non è elencato fallisce. MPL-2.0 e LGPL sono assenti di proposito.
  `pip-audit` bloccante sul job Forge.

  Il primo run ha trovato **due vulnerabilità reali**, entrambe corrette nello
  stesso passaggio: `rustls-webpki` 0.103.10 (RUSTSEC-2026-0104, panic
  raggiungibile nel parsing delle CRL prima della verifica della firma) e
  `crossbeam-epoch` 0.9.18 (RUSTSEC-2026-0204, dereferenziazione di puntatore
  non valido), la seconda attraverso `llguidance` → `llama-cpp-2`.

  `all-features = true` non è opzionale: `MIT-0` entra da `llguidance` dietro
  una feature di `llama-cpp-2` e senza quel flag il crate non è nemmeno nel
  grafo, quindi una policy costruita sul default avrebbe avuto un buco.

  `nvidia-modelopt` è **rimosso**: dichiarato nell'extra `[distill]` di
  `forge/pyproject.toml`, sotto licenza NVIDIA e non Apache-2.0, e importato da
  zero righe di codice. Distillazione e pruning sono implementati su torch e
  transformers. Una dipendenza non usata con licenza non permissiva è solo un
  rischio senza contropartita.
  Nessun `cargo audit`, `cargo deny` o `pip-audit` gira in CI, e nulla verifica
  automaticamente la regola dichiarata obbligatoria di non introdurre dipendenze copyleft.
  Lo stato attuale è buono per diligenza manuale — le versioni bloccate in `Cargo.lock`
  sono aggiornate e senza CVE note a luglio 2026 — ma non per costruzione. Aggiungere
  `cargo deny check advisories bans licenses sources` con una allowlist di licenze
  esplicita, e `pip-audit` sul job Forge. Da verificare nello stesso passaggio:
  `nvidia-modelopt` (extra opzionale `[distill]` di `forge/pyproject.toml:38`) è distribuito
  sotto licenza NVIDIA e non è Apache-2.0 — non essendo redistribuito nel wheel il rischio è
  contenuto, ma la sua presenza merita una decisione esplicita e documentata.

- [x] **H3-D · Correggere i metadati di licenza del catalogo** *(P1)*
  In `catalog/v1/catalog.json`, `gemma-4-e4b` e `gemma-4-12b` sono dichiarati
  `Apache-2.0`: i pesi Gemma sono distribuiti sotto i *Gemma Terms of Use* di Google, che
  includono una politica d'uso vietato e obblighi sulle redistribuzioni, e non sono
  interscambiabili con Apache-2.0. Analogamente `deepseek-coder-v2-lite` è marcato `MIT`,
  licenza che copre il codice DeepSeek ma non i pesi, soggetti al *DeepSeek License
  Agreement*. La verifica merita rigore perché il progetto esclude deliberatamente Llama dal
  catalogo per un obbligo di branding: lo stesso criterio va applicato agli altri fornitori.
  Correggere le tre stringhe e aggiungere al catalogo un campo che distingua licenza del
  codice e licenza dei pesi.

- [ ] **H3-E · Riproducibilità della pipeline Forge** *(P2)*
  `forge/pyproject.toml:18-31` dichiara tutte le dipendenze con `>=` e senza limite
  superiore, non esiste lockfile, e `it-core-news-lg` è installato da una URL GitHub senza
  hash (`:49`). Due run della stessa pipeline a distanza di settimane possono usare versioni
  diverse di torch e transformers. È un prerequisito della Definition of Done di
  `forge-research-roadmap.md` («Reproducible: single command + versioned config») e della
  documentazione tecnica che il capo IV dell'AI Act richiede al fornitore: senza lockfile non
  c'è nulla da attestare. Aggiungere un lockfile (`uv.lock` o `requirements.lock` generato
  da `pip-compile`), limiti superiori sulle major, e l'hash del wheel spaCy.

- [x] **H3-F · Test di integrazione HTTP** *(P2)*
  **Chiusa il 30 luglio 2026.** `api::http_tests`: il router viene avviato su
  `127.0.0.1:0` e i test parlano HTTP vero attraverso l'intera pila di
  middleware, non chiamano gli handler. La differenza non è di stile:
  l'allowlist legge l'indirizzo del peer, e un handler testato in isolamento
  non ce l'ha. Porta effimera perché girano in parallelo a tutto il resto
  della suite.

  Quattro test, tutti su comportamento che si è rotto davvero: `/api/tags` e
  `/v1/models` devono elencare un modello che sta nello store e non è
  caricato (la forma esatta della #294 — un modello che `/v1/models` non
  nomina non è selezionabile da un editor, per quanto giri bene); un nome di
  modello sbagliato deve tornare un errore che lo nomina e non un 500; e
  `/api/version` deve rispondere con lo slot vuoto, che è come `serve` parte
  sempre.

  Nessun modello caricato di proposito: l'inferenza richiede un GGUF che la
  CI non può scaricare a ogni push, e gli endpoint che rispondono senza sono
  esattamente quelli che si sono rotti. `ModelStore::at()` è nuova e sotto
  `#[cfg(test)]`: passare da `default_store()` significherebbe leggere
  `EULLM_MODELS_DIR` e `HOME`, e impostarli da un test corre contro ogni
  altro test del binario.

  Movente registrato per intero: tre difetti trovati a mano in due giorni
  vivevano tutti sul percorso `serve` — gli elenchi che ignoravano lo store,
  il banner diagnostico mai stampato, e un'intera famiglia di modelli che non
  rispondeva. Duecentodiciassette test verdi non avevano niente da dire su
  nessuno dei tre.
  Nessun test tocca `axum`, e `assert_cmd`/`predicates` sono in dev-dependencies
  (`engine/Cargo.toml:58-60`) senza alcun test che li usi. Le voci H0-A, H1-E e la forma
  delle risposte sono tutte verificabili con `tower::ServiceExt::oneshot` sul `Router`,
  senza bisogno di un modello caricato: validazione degli input, allowlist, codici di errore,
  forma di JSON/SSE/NDJSON. È il complemento naturale dei golden test già previsti da
  `0.8-D`, e va introdotto prima di quelli perché non richiede di toccare nessuna route.

- [x] **H3-G · Rimuovere il codice morto e togliere `-A dead-code`** *(P2)*
  **Chiusa il 29 luglio 2026.** Ramo rimosso (45 righe) e clippy gira senza
  soppressioni: via sia `-A dead-code` sia `-A unused-imports`, perché nessuna
  delle due era più necessaria. La variante `KvCacheType::Unknown` resta perché
  appartiene al crate `llama-cpp-2` vendored, non a noi; nostro era solo il
  ramo che la gestiva. I tre lettori dell'audit trail senza chiamante hanno un
  `#[allow(dead_code)]` puntuale che nomina H3-J.

  Ha ripagato nello stesso commit: togliere la soppressione ha fatto emergere
  una variabile resa morta dal refactor del banner di H3-N, scritta pochi
  minuti prima. È esattamente il caso che l'item descriveva — non il peso del
  codice morto, ma il fatto che il prossimo orfano non si sarebbe notato.
  Il ramo di fallback "mixed TQ" (`scheduler.rs:694-751`, ~55 righe) opera su
  `KvCacheType::Unknown(k)` con `k != v`, una condizione che `parse_cache_type`
  (`inference/mod.rs:277-290`) non può produrre: è residuo dell'integrazione TurboQuant
  rimossa in v0.5.8 ed è irraggiungibile. Sopravvive perché la CI passa `-A dead-code` a
  clippy (`ci.yml:75`). Il problema non è il peso, ma il fatto che la prossima funzione
  orfana non verrà notata: rimuovere il ramo e togliere la soppressione globale,
  riabilitandola solo con `#[allow]` puntuali dove serve davvero.

- [x] **H3-H · Condensare le opzioni di runtime in una struct condivisa** *(P2)*
  **Chiusa il 29 luglio 2026.** `RuntimeOpts` con `#[derive(clap::Args)]`,
  flattenata in `Run` e in `Serve`: 20 flag condivisi dichiarati una volta sola.
  Ogni braccio del match fa un `let RuntimeOpts { .. } = opts;` che ri-lega i
  nomi che il corpo già usava, quindi sotto quella riga non è cambiato niente.

  Sono 21 flag, non 20: la prima stesura teneva `batch_size` fuori trattando la
  differenza 1/8 come voluta. Non lo è. Chi avvia `serve` senza pensarci si
  ritrova la KV divisa per otto — 512 token per richiesta col contesto di
  default — e lo scopre da una risposta troncata a metà, riportata come
  `done_reason="length"`, che non punta a nessuna flag perché nessuna flag è
  stata passata. Il default è 1 su entrambi i comandi e la concorrenza si
  chiede. Il warning dello scheduler resta: scegliere 8 senza alzare
  `--ctx-size` produce esattamente lo stesso troncamento, solo che ora è una
  scelta.

  Restano fuori `--fit`/`--fit-strict`, che scelgono il numero di layer contro
  la VRAM libera *prima* del load, mentre `serve` carica dentro
  `api::swap_model` che non ha quel passaggio. Esporli lì significherebbe
  accettarli e non fare niente, che è peggio che non offrirli: cablare
  l'auto-fit nello swap è lavoro a sé.

  Effetto collaterale rimosso nello stesso passaggio: i tre `Commands::Run { }`
  scritti a mano per gli esiti del picker, che ripetevano tutti i 27 default a
  mano — una quarta copia della stessa lista, che un default cambiato
  nell'attributo `#[arg]` avrebbe lasciato sbagliata in silenzio. Ora è
  `picker_run()`, che chiede i default a clap parsando `eullm run -- <model>`.
  Verificato che i 25 valori dei literal coincidevano con i default di clap,
  quindi la rimozione non cambia comportamento.

  Due test nuovi: uno confronta l'intera `RuntimeOpts` fra i due comandi, uno
  passa gli stessi flag a entrambi e confronta il risultato. Il `--help` reale
  dei due comandi differisce solo per `--cli`, `--fit`, `--fit-strict`,
  `--image`, `--no-ui` (run) e `--ui` (serve).

  **La regola di parità obbligatoria nel `CLAUDE.md` è ora superflua per i
  campi in `RuntimeOpts`**, che è quello che l'item chiedeva: la divergenza è
  impossibile per costruzione invece che vietata per convenzione. Va riscritta
  per dire questo, e per coprire il caso che resta scoperto — un campo tenuto
  deliberatamente fuori dalla struct.
  `main.rs` è 3.178 righe e i 22 campi di `Commands::Run` sono replicati a mano tre volte
  (`main.rs:540-595, 624-651`): aggiungere una flag richiede quattro modifiche coordinate.
  Il `CLAUDE.md` documenta questo esatto errore come già avvenuto in produzione
  (`cache_type_k`/`gpu_layers` presenti su `Run` e non su `Serve`) e ne ha fatto una regola
  obbligatoria — ma la regola è un presidio umano su un problema strutturale. Una
  `struct RuntimeOpts` con `#[command(flatten)]`, condivisa da `Run` e `Serve`, rende la
  divergenza impossibile per costruzione invece che vietata per convenzione, e la regola
  diventa superflua.

- [ ] **H3-I · Unificare i due cataloghi** *(P2)*
  Il catalogo dell'Engine (`catalog/v1/catalog.json`) contiene 22 modelli upstream ed è
  JSON versionato; quello dell'Hub è hardcoded in Rust (`hub/src/main.rs:78-151`) ed elenca
  sette modelli `eullm/*` che non esistono ancora, con dimensioni e modello sorgente
  dichiarati (fra cui `code-eu-14b` da `deepseek-ai/DeepSeek-V3`, che non è un percorso di
  compressione plausibile). Quando i modelli verticalizzati esisteranno serviranno due
  aggiornamenti manuali coordinati. Unificare su `catalog/v1/catalog.json`, con l'Hub che lo
  serve e l'Engine che lo consuma, e distinguere nello schema i modelli disponibili da
  quelli annunciati.

- [ ] **H3-J · Igiene dell'audit trail** *(P2, dopo H0-B)*
  `AuditLogger::new()` è costruito a ogni richiesta e riapre il file ogni volta
  (`routes.rs:580, 631, 804` → `audit/mod.rs:120-138`), senza `fsync`, senza rotazione e
  con i permessi di default; `read_all` e `count` caricano l'intero file in memoria
  (`:141-165`). Tenere un handle aperto con `fsync` periodico, impostare permessi
  restrittivi sulla directory, aggiungere rotazione e rendere `count` incrementale.

- [x] **H3-M · Un tag di release non deve poter partire da un `main` rosso** *(P1)*
  `release-engine.yml` si attiva su `push: tags: [engine-v*, EuLLM-v*]` e non ha
  alcuna dipendenza dallo stato di `ci.yml` (nessun `workflow_run`, nessun
  controllo di check suite). Un tag su un commit la cui CI è fallita — o non è
  ancora finita — produce comunque 13 binari pubblicati sulla release page.
  Questo è il solo punto in cui "scoprire un problema prima o dopo il merge"
  cambia davvero qualcosa: normalmente un `main` rosso si corregge con un
  commit, qui diventa un artefatto scaricabile. Aggiungere un job iniziale che
  verifichi la conclusione della CI sul commit del tag e fermi il workflow se
  non è `success`, oppure un `workflow_dispatch` con conferma esplicita per i
  casi di emergenza. Costo: zero run aggiuntivi — è un controllo di API, non una
  compilazione. Alternativa a costo zero e zero codice: non taggare mai prima di
  aver visto la spunta verde su `main`.

- [x] **H3-K · Completare `SECURITY.md`** *(P3)*
  Il file è ancora il template GitHub non compilato: la tabella delle versioni supportate
  elenca `5.1.x` e `4.0.x`, che non esistono, e la sezione di segnalazione contiene le
  istruzioni segnaposto invece di un canale di contatto. Un progetto che pubblica binari e
  punta a un pubblico enterprise deve avere un canale di disclosure dichiarato e una policy
  sulle versioni supportate coerente con lo schema di release effettivo.

- [x] **H3-L · Header di sicurezza sulla chat UI** *(P3)*
  `ui/mod.rs:58-79` serve gli asset con `Content-Type` e `Cache-Control` e nulla più. La
  disciplina di escaping in `app.js` è corretta — `renderContent` neutralizza l'HTML prima
  delle trasformazioni Markdown, incluse le sezioni di reasoning — ma un
  `Content-Security-Policy` restrittivo (la UI non carica nulla dall'esterno per progetto,
  quindi `default-src 'self'` è compatibile) più `X-Content-Type-Options: nosniff` sono una
  difesa in profondità a costo nullo che rende la proprietà "zero risorse esterne"
  verificabile dal browser invece che solo dichiarata.

- [x] **H3-O · Compilare dai sorgenti era rotto per chiunque seguisse il
  README** *(P1)*
  Segnalato dalla issue #286, da openSUSE Tumbleweed: la build moriva su
  `'llama.cpp/include/llama.h' file not found` dentro bindgen. Il primo sospetto
  di chi segnalava è stata la distribuzione, il che è ragionevole: quell'errore
  sembra un toolchain rotto.

  Era il nostro README. llama.cpp è un **submodule** dall'8 luglio, e il README
  diceva `git clone` senza `--recursive` in **due** punti. Quindi da tre
  settimane ogni compilazione da sorgente fatta seguendo le istruzioni
  falliva. La CI non se n'è accorta perché il workflow fa
  `submodules: recursive`: l'unico percorso rotto era quello che nessuno di noi
  percorre. In più gli archivi `Source code (zip/tar.gz)` allegati a ogni
  release **non possono funzionare per definizione**, perché GitHub li genera
  senza il contenuto dei submodule.

  Chiuso su entrambi i fronti: il README clona con `--recursive` e spiega come
  sistemare un clone già fatto, e `build.rs` controlla i sorgenti prima di
  partire e stampa il comando che risolve, invece di lasciare che il sintomo
  arrivi minuti dopo travestito da errore di compilatore.

- [x] **H3-P · Il tag di una release poteva precedere il suo bump di versione**
  *(P1)*
  La v0.6.43 è stata taggata un merge prima del commit che alzava la versione.
  Risultato: nove binari pubblicati che rispondono `0.6.42` a `-V`, un
  `CHANGELOG.md` senza la sezione di quella versione, e la correzione della
  build da sorgente (H3-O) rimasta fuori dalla release che avrebbe dovuto
  contenerla. **Niente è fallito e la pagina della release sembrava normale**:
  lo scarto era visibile solo eseguendo un artefatto scaricato.

  Chiuso con `require_version_match` in `release-engine.yml`, che confronta il
  tag con `engine/Cargo.toml` e blocca `release` — non le build, che da un
  commit mal taggato non costano niente. Gira in pochi secondi, quindi un tag
  sbagliato diventa rosso mentre le build lunghe stanno ancora partendo e fa in
  tempo a essere cancellato e ripushato.

  Un tag già pubblicato **non si sposta**: la pagina e i suoi checksum sono già
  in mano a chi ha scaricato. Si rilascia la patch successiva e si scrive nel
  changelog cosa conteneva davvero quella sbagliata.

- [x] **H3-Q · La release pubblicava una lista scritta a mano, non ciò che
  aveva costruito** *(P1)*
  La v0.6.47 ha pubblicato nove binari su dieci. `build-vulkan` è andato a
  buon fine, il suo artefatto è stato scaricato, il suo checksum è finito in
  `checksums.txt`, e il file non è mai stato allegato: l'elenco dei file da
  pubblicare era scritto a mano e per quello nuovo nessuno aveva aggiunto la
  riga. In più `fail_on_unmatched_files: false` — che esiste perché una build
  fallita non affondi la release — rendeva l'omissione **silenziosa per
  costruzione**.

  Ora il passo di pubblicazione allega `artifacts/*/*`. Ogni job carica un
  file dentro una directory che porta il suo nome, quindi il glob **è**
  l'insieme di ciò che è stato costruito, e aggiungere un job dimenticando la
  riga di pubblicazione è un errore che il workflow non può più commettere.

  Indizio da ricordare: `checksums.txt` viene generato camminando le directory
  degli artefatti, quindi elencava il binario mancante **entrambe le volte**
  (988 byte → 1100). Quando una release sembra sbagliata, confrontare il file
  dei checksum con gli asset allegati risponde separatamente a "è stato
  costruito?" e "è stato pubblicato?".

  **Contesto, perché la lezione vera è cumulativa.** In un solo pomeriggio tre
  release hanno annunciato qualcosa che non contenevano: la 0.6.43 il proprio
  numero di versione, la 0.6.46 un binario che non era stato compilato, la
  0.6.47 un binario compilato e non allegato. Tre cause diverse, un'unica
  forma: **il meccanismo di rilascio si fidava della memoria di chi lo usava**.
  Le tre correzioni sono altrettanti controlli — `require_version_match`, il
  `workflow_dispatch` che valida una toolchain senza spendere una release, e
  questo glob. Nessuna delle tre è stata trovata da una revisione: tutte e tre
  guardando cosa conteneva davvero il tag prima di scrivere "fixed in vX".

- [x] **H3-N · Le diagnostiche di piattaforma mancano su `eullm serve`** *(P2)*
  **Chiusa il 29 luglio 2026.** Banner estratto in `src/banner.rs`
  (`ModelBanner`), chiamato da `cmd_run` allo startup e da `api::swap_model`
  dopo ogni caricamento riuscito. `serve` parte senza modello, quindi stampa
  subito gli endpoint e il blocco diagnostico a ogni load: uno swap è raro e
  costoso e può cambiare `ctx_size` e i tipi di KV, quindi ristampare vale più
  delle righe che costa. `ModelReadyInfo` ha ora `Default`, e lo zero è letto
  come «non noto»: il banner omette la riga KV e l'hint sul contesto
  addestrato invece di stampare uno 0 che sembra un dato.
  Il banner con `GPU backend`, `CPU features`, `GPU layers`, `Context`, `KV cache`
  e `Threads` è stampato solo da `cmd_run` (`main.rs:1823-1882`); `cmd_serve`
  stampa sei righe e nessuna di queste. La riga `CPU features` è stata aggiunta
  in 0.6.33 proprio per diagnosticare l'issue #140, ma chi testa attraverso
  `serve` — cioè qualunque harness automatizzato, e chiunque usi l'engine come
  backend — non la vede mai. Trovato scrivendo `tools/smoke_test.py`, che ha
  dovuto avviare un secondo processo `run` solo per catturarla. È la stessa
  classe di problema della regola obbligatoria di parità dei flag `run`/`serve`
  nel `CLAUDE.md`, applicata all'output diagnostico invece che alla
  configurazione: estrarre la stampa del banner in una funzione condivisa e
  chiamarla da entrambi i comandi, emettendola dopo il primo caricamento del
  modello nel caso di `serve` (che parte senza modello).

- [ ] **H3-S · Backend GPU caricati a runtime invece che compilati nel binario** *(nice to have)*
  Oggi pubblichiamo binari separati per backend (CPU, CUDA, Vulkan) e, dentro
  quello CUDA, tre architetture compilate insieme nello stesso file (`sm_86`,
  `sm_89`, `sm_120` — Ampere/Ada/Blackwell), il che lo porta a ~900 MB. Un
  confronto diretto con una build di llama.cpp compilata per una sola
  architettura (~500 MB, solo `sm_120`) ha chiarito che la differenza è
  interamente quella scelta di copertura, non spreco nostro — ma resta il
  motivo per cui l'idea vale la pena registrarla.

  llama.cpp offre già il meccanismo per farlo diversamente: `GGML_BACKEND_DL`
  (`ggml/CMakeLists.txt:86`) compila i backend come librerie dinamiche caricate
  a runtime tramite un vero registro (`ggml_backend_reg_*`,
  `ggml/include/ggml-backend.h`), invece di essere linkati staticamente in base
  a quale cargo feature ha compilato il binario. Un `eullm` CPU-only minimo
  potrebbe fare il discovery della GPU al primo avvio e proporre di scaricare
  solo il backend giusto per quella scheda, invece di scegliere tra binari
  interi o subire tre architetture in uno.

  È la stessa idea di B1/B2 (la strategia DLL Windows già documentata più
  sopra, oggi non urgente) generalizzata a ogni piattaforma e resa automatica.
  Il costo reale non è tecnico ma di fiducia: un binario firmato con un
  checksum è una cosa, un programma che scarica ed esegue codice nativo in
  base a cosa trova sulla macchina è un'altra — servirebbe verifica del
  checksum sul backend scaricato e probabilmente una conferma esplicita, non
  un download silenzioso. Tocca anche `llama-cpp-sys-2/build.rs`,
  l'inizializzazione del backend in `inference/mod.rs`, e l'intera matrice di
  `release-engine.yml` (i backend andrebbero pubblicati come artefatti a
  parte). Prima di impegnarsi nel resto, verificare che `GGML_BACKEND_DL`
  compili pulito sul nostro commit pinnato è il primo passo, non tutto il
  lavoro insieme.

- [ ] **H3-R · Bump di `llama.cpp` a `b10200`: portati i 3 cambi API a mano,
  in attesa di validazione su hardware reale** *(P2)*
  Primo tentativo (31 luglio 2026, mattina): spostare il solo pin del
  submodule da `9e3b928` (7 giugno) a `5f55650` (30 luglio, tag `b10200`)
  fa fallire `cargo build` subito, con 9 errori — tre cambi reali nell'API C:
  `llama_model_params` non ha più i campi `use_mlock`/`use_mmap` (sostituiti
  da un enum unico `load_mode`, commit a monte *"args: refactor mlock/mmap/
  directio into load-mode (#20834)"*, 23 luglio); `mtmd_input_text` ora
  richiede un campo `text_len`; l'helper che caricava un bitmap da file/buffer
  restituiva un puntatore grezzo, ora restituisce una struct wrapper in stile
  RAII (`mtmd_helper_bitmap_wrapper`, con un campo `video_ctx` per un supporto
  video che il nostro build non compila). Il pin è stato riportato a `9e3b928`
  nello stesso momento, e il tentativo è stato loggato senza altre azioni.

  **Verifica che ha cambiato il piano**: clonando `eullm/llama-cpp-rs` per
  vedere se una versione più recente del wrapper già parlava con l'API nuova,
  il pin del submodule `llama.cpp` sul `main` di quel repository (commit
  `918853e`, 28 luglio) risulta **ancora fermo a `9e3b928`** — lo stesso
  identico commit da cui EuLLM è partito. `utilityai/llama-cpp-rs` non ha
  ancora bumpato oltre il nostro pin. Non esiste quindi, oggi, nessun commit
  upstream da cui ri-vendorizzare che risolva il problema: aspettare non
  avrebbe funzionato, e "ri-vendorizzare" nel senso stretto (copiare un
  albero più recente) non era un'opzione disponibile.

  **Quello che si è fatto invece**, lo stesso giorno, senza fretta essendo la
  release attuale stabile: portare a mano i 3 cambi nel nostro `llama-cpp-2`
  vendorizzato, restando sulla stessa base upstream (`utilityai/llama-cpp-rs
  main @ 8625c7c4`) con patch locali documentate inline (vedi il commento in
  cima a `llama-cpp-2/Cargo.toml`):
  - `model/params.rs`: `use_mmap()`/`use_mlock()`/`with_use_mmap()`/
    `with_use_mlock()` ora leggono/scrivono `load_mode` tramite una funzione
    `load_mode_from_flags`, mantenendo l'API pubblica (le due flag booleane)
    identica a prima — i default restano `use_mmap=true`, `use_mlock=false`
    (`LLAMA_LOAD_MODE_MMAP`, verificato contro `llama_model_default_params()`
    a monte).
  - `mtmd.rs`, `tokenize()`: aggiunto `text_len: text_cstring.as_bytes().len()`
    al costruttore di `mtmd_input_text`.
  - `mtmd.rs`, `MtmdBitmap::from_file`/`from_buffer`: il valore di ritorno
    diventa `mtmd_helper_bitmap_wrapper`; si estrae `.bitmap` (il campo
    `.video_ctx` è sempre nullo, dato che `MTMD_VIDEO` non è definito nel
    nostro `build.rs`, quindi va ignorato senza perdita di comportamento).

  Il pin del submodule è ora a `5f55650` (`b10200`). Validato finora, in
  locale, senza GPU disponibile in questo ambiente: `cargo build` e
  `cargo clippy` puliti sia con `--features multimodal` che con
  `--no-default-features`; le 50 doctest di `llama-cpp-2` passano, incluse
  quelle che affermano esplicitamente i default di `use_mmap`/`use_mlock`;
  le 225 unit test di `eullm-engine` passano.

  **Validazione su hardware reale iniziata il 3 agosto, su `rc4`**: ha trovato
  subito un gap reale nel probe di `probe_and_shrink_context` — vedi H3-T,
  già corretto nella stessa giornata. Restano da ricaricare le altre famiglie
  di modelli (incluso il template di ragionamento DeepSeek) prima di
  considerare `0.6.70` pronta per l'uscita definitiva.

- [ ] **H3-T · Il probe del contesto non usava lo stesso sizing della vera
  richiesta multimodale** *(P2)*
  Trovato il 3 agosto 2026 testando `rc4` su hardware reale: un modello
  vision 12B Q8 caricava pulito a `--ctx-size 4096` (il probe di
  `probe_and_shrink_context` passava), e poi il primo messaggio con
  un'immagine allegata falliva con lo stesso errore di allocazione che il
  probe esiste apposta per intercettare — `could not allocate a context of
  4096 tokens ... Its KV cache alone needs about 3072 MiB`.

  Causa: `generate_multimodal` dimensiona `n_batch`/`n_ubatch` diversamente
  dal resto — un encoder visivo usa attenzione non causale, quindi l'intera
  immagine deve entrare in un solo micro-batch (`n_ubatch >= image_tokens`,
  altrimenti `GGML_ASSERT` va in crash). Questo sizing (`mm_batch`, funzione
  `multimodal_batch_size`) era più grande di quello usato dal probe a
  caricamento (`build_ctx_params`, che deriva `n_ubatch` da `config.n_batch`
  limitato a 1024). Un buffer di calcolo più grande serve più VRAM: il probe
  provava una richiesta più leggera di quella che una vera immagine avrebbe
  fatto, quindi "ci sta" a caricamento non garantiva "ci sta" al primo
  messaggio con immagine — esattamente il divario che la validazione su
  hardware reale, e non la build pulita, doveva scoprire.

  Fix: `multimodal_batch_size` estratta come funzione condivisa; il probe in
  `probe_and_shrink_context` ora, quando `config.mmproj_path` è impostato,
  costruisce i parametri di prova con lo stesso `mm_batch` di
  `generate_multimodal`, non più con i parametri di testo semplice. Verificato
  in locale (senza GPU in questo ambiente): `cargo build`/`clippy` puliti con
  e senza `--features multimodal`, le 225 unit test passano. Da confermare su
  hardware reale con lo stesso modello che ha esposto il problema, in `rc5`.

  **Confermato su hardware reale in `rc5`, il 3 agosto**: niente più crash al
  primo messaggio con immagine, a `--ctx-size 4096` e a `16384` — in entrambi i
  casi il probe scende in modo consistente fino a `1024`, che è quindi il vero
  tetto trovato, non un artefatto del valore di partenza. Ma `1024` si è
  rivelato troppo stretto per una chat multi-turno con immagine (risposte
  tagliate a metà frase per esaurimento del contesto condiviso — vedi i log
  con `num_predict capped`), e ha esposto un **secondo problema**, distinto da
  quello che questa voce descriveva in origine: `mm_batch` di default seguiva
  `--n-batch` (2048, il batch di testo), non la reale dimensione in token
  dell'immagine (~266 per una slice di Gemma 4, verificato dal log
  `n_tokens_batch = 266` dello stesso caricamento). Un buffer di calcolo
  dimensionato per 2048 token quando ne servono ~266 schiacciava la KV cache
  ben oltre il necessario. Corretto in `rc6`: `multimodal_batch_size` non
  dipende più da `config.n_batch` — usa `EULLM_IMAGE_MAX_TOKENS` se impostata,
  altrimenti il pavimento di 512 già previsto.

  **Bug simmetrico trovato il 4 agosto testando `rc7`**: il fix di `rc6` ha
  corretto il probe per il caso immagine ma rotto quello di solo testo. Un
  modello caricato con mmproj riceve anche messaggi senza immagine, che
  passano da `generate`/`generate_streaming` — percorso che usa
  `--n-batch` limitato a 1024, non il batch piccolo dell'immagine. Il probe
  (dopo `rc6`) validava solo il caso immagine (batch 512), più leggero;
  avviato con `--ctx-size 65536`, si è ridotto pulito a 4096, ma il primo
  messaggio di solo testo (senza foto allegata) è fallito con lo stesso
  errore di allocazione che il probe doveva intercettare — un messaggio
  successivo con foto invece è passato. Corretto in `rc8`: il probe ora
  prende il **massimo** tra i due batch possibili (quello del testo normale
  e quello dell'immagine), non solo quello dell'immagine — copre entrambi i
  casi che lo stesso modello caricato può davvero servire.

  Resta da confermare su hardware reale quanto sale il tetto rispetto ai
  16k che `llama-cli`/`llama-server` gestiscono sullo stesso modello —
  divario non ancora spiegato del tutto (vedi la sessione di debug
  comparativo di prima di questo bump, mai conclusa).

- [ ] **H3-U · Il template di chat era scelto per nome, non letto dal GGUF —
  ora si usa il vero template quando c'è** *(P2)*
  Trovato il 3 agosto 2026, confrontando le risposte di eullm con quelle di
  `llama-server` sullo stesso `gemma-4-12b-q8`: il template di chat reale del
  file — quello nei metadati del GGUF stesso — è un formato a canali con
  supporto per il tool-calling (`<|turn|>`, `<|channel|>thought…`), non ha
  niente a che vedere con `<start_of_turn>`/`<end_of_turn>` di Gemma. La
  nostra rilevazione (`ChatTemplate::detect`, per nome del modello) sceglieva
  comunque il template Gemma, costruendo un prompt nella forma sbagliata. Il
  modello rispondeva comunque — un LLM tollera un prompt leggermente fuori
  formato — ma non nel modo per cui è stato istruito, ed è la stessa causa
  dietro le fughe di marcatori `<|channel|>`/`<|message|>` che i filtri
  Harmony (aggiunti in 0.6.69) tamponavano già senza risolvere.

  Verificato leggendo il codice sorgente di llama.cpp (non ipotizzato):
  `common_chat_templates_init` in `common/chat.cpp` legge sempre per prima
  cosa il template incorporato nel GGUF (`llama_model_chat_template`) e lo
  applica con il proprio motore Jinja (`minja`) — il fallback a un ChatML
  hardcoded scatta solo se il file non ne ha nessuno. `llama-server` non fa
  nessuna distinzione per nome di modello: usa sempre quello che il file
  dichiara di essere. C'è perfino un formato registrato esplicitamente,
  `COMMON_CHAT_FORMAT_PEG_GEMMA4`, a conferma che questo non è un file
  etichettato male ma un formato che llama.cpp riconosce di suo.

  Fix: nuova funzione C `llama_rs_apply_chat_template` in
  `llama-cpp-sys-2/wrapper_common.cpp`, che espone
  `common_chat_templates_init`/`common_chat_templates_apply` con la stessa
  convenzione (`llama_rs_status`, stringhe allocate liberate con
  `llama_rs_string_free`) degli altri wrapper già presenti. Wrapper Rust
  sicuro `LlamaModel::apply_jinja_chat_template` in `llama-cpp-2`, e sopra
  `InferenceEngine::apply_jinja_chat_template` in eullm, che lo usa **solo
  quando il GGUF ha davvero un template incorporato** (`was_explicit`) —
  altrimenti restituisce `None` e il chiamante ricade sui nostri template
  hardcoded, esattamente come oggi.

  **Limite di scope deliberato, non un taglio per fretta**: attivo solo in
  modalità sequenziale (`snap.engine`, quindi anche ogni modello
  multimodale) — lo scheduler a batch continuo gira il modello sul proprio
  thread dedicato e non lo espone a questo livello, quindi le richieste in
  quella modalità restano sui template hardcoded finché qualcuno non fa
  anche quel lavoro. Il testo generato dal modello non viene ripulito dal
  blocco di ragionamento (`thinking_start_tag`/`thinking_end_tag` che
  llama.cpp restituisce insieme al prompt) — quella pulizia resta ai filtri
  Harmony esistenti, non ancora collegata ai tag veri del template.

  Verificato in locale (senza GPU in questo ambiente): `cargo build`/
  `clippy` puliti con e senza `--features multimodal`, le 225 unit test
  passano. **Confermato su hardware reale il 5 agosto** per i messaggi di
  solo testo via chat web: la risposta ora mostra il blocco di ragionamento
  reale del template, coerente con quanto osservato su `llama-server`.

  **Incoerenza trovata lo stesso giorno**: il fix copriva solo `routes.rs`
  (chat web/API) — `eullm run --cli` continuava a costruire il prompt col
  solo template hardcoded, invariato, quindi la stessa domanda allo stesso
  modello dava risposte diverse a seconda della porta usata per chiederla.
  Corretto con `build_cli_prompt` in `main.rs`, che rispecchia esattamente
  `api::routes::build_chat_prompt` (stessa politica: template dinamico solo
  in modalità sequenziale, altrimenti il fallback hardcoded).

  Resta da confermare su hardware reale che non cambi comportamento sugli
  altri modelli già validati (DeepSeek R1, Qwen3) se anche i loro GGUF
  portano un template incorporato che ora prende il posto di quello scritto
  a mano per loro — e resta non collegata la pulizia della risposta dal
  blocco di ragionamento (`thinking_start_tag`/`thinking_end_tag`), ancora
  affidata solo ai filtri Harmony esistenti.

  **Aggiornamento 6 agosto**: il sospetto sopra sembrava confermato — test su
  hardware reale con DeepSeek-R1-Distill-Qwen-14B, Qwen2-VL-2B e la variante
  da 7.1GB di gemma-4-e4b hanno prodotto risposte a vanvera via chat web
  (DeepSeek: un'intera derivazione di calcolo integrale invece di rispondere
  "come ti chiami"; gli altri due: allucinazioni di identità tipo "sono il
  software OpenOffice.org calc"). La causa reale però non era il template
  dinamico: la stessa identica domanda sullo stesso modello via `--cli` ha
  risposto correttamente (vedi H3-W). Il meccanismo del template dinamico —
  la parte che questa voce copre — è quindi scagionato per DeepSeek R1 da un
  confronto diretto sullo stesso modello. Resta comunque da confermare via
  `--cli` anche su Qwen2-VL-2B e gemma-4-e4b (finora testati solo via web,
  dove il vero colpevole — il messaggio di sistema di default — era
  comunque presente), e resta da validare Qwen3 come già annotato sopra.

- [x] **H3-V · `probe_and_shrink_context` si ferma al primo tentativo che
  passa, non cerca il vero massimo** *(nice to have)*
  Osservato il 4 agosto 2026 testando `rc7` con un valore deliberatamente
  non tondo: `--ctx-size 8112` → fallisce → dimezzato esattamente a `4056`
  (8112 / 2) → questo passa al primo tentativo → l'algoritmo si ferma lì.
  Non prova mai valori intermedi tra 4056 e 8112 (es. il `4096` che sappiamo
  già andare bene da un test precedente) — il numero restituito è sempre
  garantito funzionante (è una vera allocazione, non una stima), ma non è
  il tetto reale della scheda, solo il primo `requested / 2^k` che ci sta.
  Non è un bug: è esattamente il comportamento descritto nel commento della
  funzione ("halving until one fits or a floor is reached"), solo che con
  un valore di partenza non potenza di due lo si nota — con partenze tipo
  16384/8192/4096 il dimezzamento atterra sempre sugli stessi numeri già
  visti prima, mascherando quanto è grezzo il meccanismo.

  **Diventato urgente il 5 agosto** quando un secondo test ha esposto perché
  la grossolanità del dimezzamento non è solo una questione di contesto
  sprecato: su `rc8`, `--ctx-size 65536` si è ridotto pulito a `4096`
  (nessun avviso fuori dall'ordinario), e la prima richiesta reale — un
  messaggio di solo testo — ha **mandato in crash il processo** (un
  `GGML_ASSERT` di llama.cpp, non l'errore pulito che questo probe esiste
  per produrre). Rilanciando lo stesso identico comando, il probe è atterrato
  su un valore più piccolo ed è andato tutto bene — la VRAM libera
  fluttuava leggermente tra un avvio e l'altro, e `4096` ci stava per un
  margine talmente stretto da non reggere quella fluttuazione al momento
  della richiesta vera.

  Corretto in `rc9`, unendo due cambi nello stesso posto:
  - **margine di sicurezza**: dopo un'allocazione di prova riuscita,
    `gpu_free_ratio()` (nuova funzione, somma `ggml_backend_dev_memory` sui
    dispositivi di tipo GPU) controlla che resti libero almeno il 12% della
    memoria totale della scheda — non solo che l'allocazione sia riuscita.
    Sotto quella soglia il candidato viene scartato come se fosse fallito
    del tutto. 12% è il punto di mezzo dell'intervallo 10-15% indicato,
    scelto perché ha fatto sparire il crash nella pratica, non calcolato da
    una formula.
  - **raffinamento fine**: una volta trovato un valore che passa dimezzando
    (fase grezza, invariata), si risale a passi di 1024 token verso l'alto,
    fermandosi appena il valore successivo non passa più (allocazione o
    margine) — recuperando il terreno intermedio che il solo dimezzamento
    saltava, verificato ogni passo con una vera allocazione, mai stimato.

  Verificato in locale (senza GPU in questo ambiente): `cargo build`/
  `clippy` puliti con e senza `--features multimodal`, le 225 unit test
  passano. Da confermare su hardware reale che il crash non si ripresenti e
  che il raffinamento recuperi davvero contesto utile tra i valori dimezzati.

- [x] **H3-W · Il messaggio di sistema di default della chat web mandava in
  crisi modelli diversi da quello per cui sembrava pensato** *(P1)*
  Trovato il 6 agosto 2026, subito dopo aver reso coerenti CLI e chat web
  (H3-U): con lo stesso identico modello e la stessa identica domanda ("ciao
  come ti chiami"), `--cli` rispondeva correttamente e la chat web no, su tre
  modelli diversi — DeepSeek-R1-Distill-Qwen-14B, Qwen2-VL-2B, gemma-4-e4b
  (variante 7.1GB). Questo esclude il template dinamico come causa (H3-U):
  se fosse stato quello, `--cli` avrebbe fallito allo stesso modo, dato che
  usa la stessa identica funzione di decisione. L'unica differenza reale tra
  le due porte era il messaggio di sistema: `--cli` parte con un generico
  "You are a helpful assistant.", la chat web mandava di default un
  messaggio più elaborato (un suggerimento di formattazione LaTeX, aggiunto
  in una sessione precedente).

  Su DeepSeek-R1-Distill-Qwen-14B l'effetto era netto: invece di rispondere
  al saluto, il modello ha prodotto un'intera derivazione di calcolo
  (l'integrale di eˣsin(nx)) — nessun esempio del genere è presente nel testo
  del messaggio di sistema stesso, quindi non è un caso di "ripete quello che
  gli è stato scritto". La spiegazione più coerente: DeepSeek sconsiglia
  esplicitamente l'uso di un system prompt con i modelli R1 (la scheda del
  modello raccomanda di mettere le istruzioni nel turno utente), e un
  messaggio di sistema insolito sembra spingerli a replicare una traccia di
  ragionamento stereotipata vista in training (fortemente sbilanciata su
  matematica/codice) invece di rispondere al turno reale. Su Qwen2-VL-2B e
  gemma-4-e4b l'effetto era diverso — allucinazioni di identità — ma
  scompariva comunque disattivando il messaggio di sistema di default.

  Fix in rc11 (`engine/src/ui/app.js`): `settings.system` parte vuoto invece
  del testo LaTeX. Entrambi i percorsi di invio (`/api/chat` multimodale e
  `/v1/chat/completions`) già saltano del tutto il turno di sistema quando
  `settings.system` è vuoto (`if (settings.system) ...`), quindi il default
  diventa "nessun messaggio di sistema", non "messaggio di sistema vuoto" —
  lo stesso comportamento di fatto già in uso da `--cli`. Il suggerimento
  LaTeX resta disponibile, opt-in, da Settings.

  **Seguito in rc12**: disattivare il suggerimento di default perdeva la
  formattazione automatica delle formule per chiunque non sapesse di doverla
  riattivare a mano — segnalato subito dall'utente ("non sapendo se ci
  saranno formule o meno"). Corretto spostando `MATH_FORMAT_HINT` dal campo
  libero `settings.system` (rimasto vuoto di default, riservato a istruzioni
  scritte dall'utente e mandate come vero turno `system`) a un'aggiunta in
  coda al turno **utente** in uscita, condizionata a `settings.math` (attivo
  di default). Stesso suggerimento, stesso comportamento attivo-di-default di
  prima della regressione, ma senza reintrodurre un turno di sistema — la
  parte che effettivamente rompeva i modelli.

  Verificato in locale (senza GPU in questo ambiente): non è codice Rust,
  nessuna build da rifare. Da confermare su hardware reale che la chat web
  torni a rispondere correttamente sui tre modelli e che il suggerimento
  LaTeX in coda al turno utente non introduca a sua volta problemi.

  **Confutato su hardware reale, stesso giorno**: la teoria di rc12 (era il
  turno `system` il problema, non il contenuto) non ha retto alla prova.
  Stessa domanda ("ciao come ti chiami"), stesso modello
  (DeepSeek-R1-Distill-Qwen-14B): il log del server mostra `prompt=57
  tokens` per la richiesta web contro i `prompt=16 tokens` di `--cli` per la
  domanda identica — i 41 token in più sono `MATH_FORMAT_HINT`, quindi
  l'iniezione nel turno utente stava avvenendo come previsto. Il risultato è
  comunque un'allucinazione di identità, stavolta diversa ("Mi chiamo
  MathAI" invece di rispondere normalmente). Il fattore comune ai due
  tentativi falliti non era il ruolo del messaggio ma l'aver accodato
  un'istruzione estranea a un prompt corto e non correlato — `--cli`, che
  non manda niente del genere, risponde correttamente ogni volta.

  Corretto in rc13 rimuovendo del tutto `MATH_FORMAT_HINT` e la sua
  iniezione automatica (sia nel turno utente del ramo multimodale sia in
  quello del ramo testo). Nessun suggerimento di default in nessuna forma;
  resta disponibile solo scrivendolo a mano nel campo "System prompt" di
  Settings, inviato come vero turno `system` — comportamento invariato per
  chi lo usa già così.

  Verificato in locale (senza GPU in questo ambiente): non è codice Rust,
  nessuna build da rifare. Da confermare su hardware reale che senza alcuna
  iniezione automatica la chat web risponda come `--cli` sui tre modelli.

- [ ] **H3-X · QwQ-32B-Preview via chat web: `<|im_start|>` trapela nel
  testo e la risposta si tronca a 7 token** *(P2)*
  Trovato il 7 agosto 2026 su rc14, testando `eullm run --fit` con
  `qwq-32b` (QwQ-32B-Preview-Q4_K_M) scelto dal picker: alla domanda "ciao
  come ti chiami?" la chat web ha risposto `Mi chiamo AI.<|im_start|>` — 7
  token totali (confermato dall'audit log: `output_tokens=7`), col marker
  ChatML renderizzato come testo visibile e la generazione interrotta lì.
  Il modello si era caricato correttamente (split 43/64 layer via `--fit`,
  nessun OOM) — il problema è a valle, nella costruzione del prompt o nella
  gestione dei token di stop, non nel caricamento.

  Sospetti da verificare, in ordine: (a) QwQ-32B-Preview è un modello
  reasoning con un template proprio — il template dinamico (H3-U) potrebbe
  renderizzarlo in una forma che il modello non si aspetta, come già visto
  per gemma-4-12b-q8; (b) sul percorso dinamico `stop_sequences` è vuoto e
  lo stop dipende solo da `is_eog_token` — se il modello emette
  `<|im_start|>` (inizio turno, non fine) come testo e *poi* qualcosa lo
  ferma a 7 token, c'è anche una domanda su cosa abbia fermato la
  generazione così presto. Ispezionare il template GGUF incorporato del
  file con lo script già usato per gemma-4.

  **Isolamento già fatto (7 agosto, rc14)**: stessa domanda via `--cli`
  sullo stesso modello → risposta pulita ("Ciao! Mi chiamo AI.", 8 token,
  stop regolare, nessun marker nel testo). Il bug è quindi confinato al
  percorso chat web, esattamente come H3-W: il prompt costruito dalla UI
  (history multipla / campi extra) va guardato per primo, non il template
  in sé, che sul percorso CLI rende correttamente.
---

## Rimandi — voci già coperte dalle roadmap esistenti

Elencate qui per non riaprirle: sono già pianificate altrove, e queste note aggiungono solo
l'evidenza sul sorgente raccolta durante la revisione.

| Osservazione | Già coperta da | Nota |
|---|---|---|
| Il prefill blocca il decode delle sequenze attive (`scheduler.rs:843-1223, 1516-1532`) | `0.7-D` Mixed chunked prefill | La diagnosi in `0.7-D` è corretta e completa. Costo misurato in ordine di grandezza: `⌈prompt_tok / n_ubatch⌉` forward pass durante i quali nessun'altra sequenza avanza — ~30 pass con un prompt da 30k token |
| Split fisso del contesto e rifiuto rigido (`scheduler.rs:679, 1441`) | `0.8-A` Scheduling a budget token | Il vincolo di correttezza descritto in `0.8-A` (eviction e accounting nello stesso branch) è la parte non ovvia e va rispettato |
| Endpoint di embedding assenti | `0.8-B` Embeddings in-process | Nota: `README.md:647` afferma «same endpoints» mentre `README.md:992` li dichiara pianificati — allineare le due frasi, la prima è quella nel claim di posizionamento |
| ~~`--fit` non tiene conto di `--cpu-moe` (`fit.rs:350` usa `file_size` come proxy)~~ | `0.7-E` Auto-composizione `--fit` + `--n-cpu-moe` | **Risolto in 0.6.70-rc14**: il parser tensor-info di `0.7-E` scompone i byte per layer in expert/non-expert. `H2-H` (ricalibrare `COMPUTE_BUFFER_RESERVE_BYTES`) resta aperto, si applica invariato anche al nuovo sizer MoE-aware |
| Contesto ricreato a ogni richiesta nel percorso sequenziale (`inference/mod.rs:685, 890, 1204`) | `1.0-A` Multimodale concorrente | Rilevante solo per il percorso sequenziale/multimodale; il default `run`/`serve` usa lo scheduler |
| Speculative decoding con draft model | `1.0-C` (benchmark-gated) | Il declassamento è motivato correttamente. Confermato indipendentemente: va comunque **dopo** `0.7-D`, perché entrambi ristrutturano la forma del batch e farli nell'ordine inverso costa il doppio |
| Coda piena senza codice di errore appropriato | `0.7-C` Backpressure HTTP | H0-A è la specializzazione concreta sui due campi che oggi non hanno limiti |
| Due ricette di distillazione coesistenti; teacher dichiarato 14B nei profili e 32B nei config | `forge-research-roadmap.md` §"To fix / reconcile" | H0-D va coordinato con questa decisione |
| Compliance card dell'Hub statica | `forge-research-roadmap.md` §F2 | H1-G è il fix immediato (404 sui nomi sconosciuti); la card generata da Forge è la soluzione |

---

## Cose che funzionano e non vanno toccate

Registrate perché un refactor le metterebbe a rischio senza accorgersene.

- **Il riuso KV a due livelli** (`scheduler.rs:501-508, 858-881` match testuale esatto;
  `:408-479` LCP per token id). La scelta di confrontare il **testo** per la parte condivisa
  e riusare i token già noti aggira l'instabilità di ri-tokenizzazione BPE al confine invece
  di limitarsi a rilevarne le conseguenze. Il commento a `:490-500` spiega perché: leggerlo
  prima di modificare qualunque cosa in quest'area.
- **La catena di fallback del prefill** (`:1001-1065`): riuso → checkpoint restore →
  re-prefill completo. Nel caso peggiore paga il comportamento vecchio e provato, mai un
  fallimento della richiesta. Invariante da preservare.
- **La verifica del digest sui pull da catalogo** (`registry/mod.rs:113-136`): il controllo
  avviene sul file `.part` **prima** del rename, quindi un download manomesso non raggiunge
  mai la destinazione. Tutte le 22 voci del catalogo hanno un digest popolato. Da estendere
  ai pull fuori catalogo (`main.rs:936, 1040` passano `None`) registrando la revisione HF
  risolta nel manifest — `resolve/main` (`registry/mod.rs:357`) non pinna nulla.
- **`is_valid_model_slug` + `canonicalize` nell'Hub** (`hub/src/main.rs:279-296, 356-365`,
  con test a `:417-443`): difesa a due livelli fatta correttamente. È il modello da riusare
  per H1-D.
- **`sanitize_for_log`** (`audit/mod.rs:60-62`): difesa corretta e testata contro il log
  forging tramite il nome modello. Da applicare a ogni nuovo campo controllato dal client
  che finisca in una riga di tracing testuale.
- **Il parser GGUF di `fit.rs`** (`:70-154`): ogni lettura bounds-checked, nessun panic
  possibile su input troncato o malformato. È il riferimento per H2-F.
