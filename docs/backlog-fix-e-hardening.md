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

- [ ] **H2-D · Unificare stop sequence e filtri tra i due percorsi di inferenza** *(P1)*
  L'hold-back buffer e i `filter_sequences` esistono solo nello scheduler
  (`process_piece`, `scheduler.rs:1588-1628`). Il percorso sequenziale e quello multimodale
  (`inference/mod.rs:1047-1085, 1357-1387`) confrontano `full_output.ends_with(s)` per
  token, quindi una stop sequence a cavallo di due token perde testo, e
  `DEFAULT_HARMONY_FILTERS` non è applicato affatto: gli artefatti `<|channel>…` che i
  filtri esistono per rimuovere restano visibili all'utente in modalità multimodale.
  Estrarre `process_piece` in un helper condiviso e usarlo in tutti e tre i loop di decode.
  Sinergia con H2-E: entrambe le voci sono conseguenze della stessa duplicazione.

- [ ] **H2-E · Estrarre la costruzione della catena di sampling** *(P2)*
  La catena è duplicata in quattro punti (`scheduler.rs:929-966`;
  `inference/mod.rs:781-812, 1000-1031, 1313-1341`) e la divergenza è già iniziata: il
  fallback del seed differisce (`seq_id` contro `1234` hardcoded) e i filtri sono assenti in
  tre copie su quattro. Ogni nuovo sampler andrà aggiunto quattro volte. Estrarre
  `fn build_sampler(&LlamaModel, &GenerateRequest, seed) -> LlamaSampler`.

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

- [ ] **H2-I · Un modello inesistente deve dare 404, non 500** *(P2)*
  Emerso testando il binario pubblicato di v0.6.35: una richiesta con
  `{"model":"non-esiste"}` restituisce **500** con `Failed to load model ...`
  (`api/routes.rs:171-180` — l'errore di `swap_model` è mappato su
  `INTERNAL_SERVER_ERROR` indipendentemente dalla causa). Chiedere un modello che
  non c'è è un errore del client, non del server: Ollama risponde 404, e un 5xx
  induce i client con retry automatico a riprovare una richiesta che non potrà
  mai riuscire. Distinguere "non trovato" (404) da un fallimento reale di
  caricamento — VRAM insufficiente, GGUF corrotto — che resta 500. Comporta far
  ritornare a `resolve_model` un errore tipizzato invece di una `String`.

- [ ] **H2-J · Con `think: false` un `</think>` finisce nel testo visibile** *(P2)*
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

- [ ] **H3-B · Il mirror non deve poter sovrascrivere i tag** *(P1)*
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

- [ ] **H3-C · Controlli automatici su dipendenze e licenze** *(P1)*
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

- [ ] **H3-F · Test di integrazione HTTP** *(P2)*
  Nessun test tocca `axum`, e `assert_cmd`/`predicates` sono in dev-dependencies
  (`engine/Cargo.toml:58-60`) senza alcun test che li usi. Le voci H0-A, H1-E e la forma
  delle risposte sono tutte verificabili con `tower::ServiceExt::oneshot` sul `Router`,
  senza bisogno di un modello caricato: validazione degli input, allowlist, codici di errore,
  forma di JSON/SSE/NDJSON. È il complemento naturale dei golden test già previsti da
  `0.8-D`, e va introdotto prima di quelli perché non richiede di toccare nessuna route.

- [ ] **H3-G · Rimuovere il codice morto e togliere `-A dead-code`** *(P2)*
  Il ramo di fallback "mixed TQ" (`scheduler.rs:694-751`, ~55 righe) opera su
  `KvCacheType::Unknown(k)` con `k != v`, una condizione che `parse_cache_type`
  (`inference/mod.rs:277-290`) non può produrre: è residuo dell'integrazione TurboQuant
  rimossa in v0.5.8 ed è irraggiungibile. Sopravvive perché la CI passa `-A dead-code` a
  clippy (`ci.yml:75`). Il problema non è il peso, ma il fatto che la prossima funzione
  orfana non verrà notata: rimuovere il ramo e togliere la soppressione globale,
  riabilitandola solo con `#[allow]` puntuali dove serve davvero.

- [ ] **H3-H · Condensare le opzioni di runtime in una struct condivisa** *(P2)*
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

- [ ] **H3-N · Le diagnostiche di piattaforma mancano su `eullm serve`** *(P2)*
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

---

## Rimandi — voci già coperte dalle roadmap esistenti

Elencate qui per non riaprirle: sono già pianificate altrove, e queste note aggiungono solo
l'evidenza sul sorgente raccolta durante la revisione.

| Osservazione | Già coperta da | Nota |
|---|---|---|
| Il prefill blocca il decode delle sequenze attive (`scheduler.rs:843-1223, 1516-1532`) | `0.7-D` Mixed chunked prefill | La diagnosi in `0.7-D` è corretta e completa. Costo misurato in ordine di grandezza: `⌈prompt_tok / n_ubatch⌉` forward pass durante i quali nessun'altra sequenza avanza — ~30 pass con un prompt da 30k token |
| Split fisso del contesto e rifiuto rigido (`scheduler.rs:679, 1441`) | `0.8-A` Scheduling a budget token | Il vincolo di correttezza descritto in `0.8-A` (eviction e accounting nello stesso branch) è la parte non ovvia e va rispettato |
| Endpoint di embedding assenti | `0.8-B` Embeddings in-process | Nota: `README.md:647` afferma «same endpoints» mentre `README.md:992` li dichiara pianificati — allineare le due frasi, la prima è quella nel claim di posizionamento |
| `--fit` non tiene conto di `--cpu-moe` (`fit.rs:350` usa `file_size` come proxy) | `0.7-E` Auto-composizione `--fit` + `--n-cpu-moe` | Il parser tensor-info previsto da `0.7-E` risolve anche questo. Coordinare con H2-A e H2-H, che toccano lo stesso sizer |
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
