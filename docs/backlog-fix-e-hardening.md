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

## H0 — Bloccanti

**Gate di uscita:** nessun componente perde silenziosamente dati che dichiara di gestire;
nessun campo ricevuto dalla rete raggiunge una decisione di allocazione senza limiti.

- [ ] **H0-A · Limiti sugli override ricevuti nel body delle richieste** *(P0)*
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

- [ ] **H0-B · L'audit trail deve rispettare `EULLM_AUDIT_DIR`** *(P0)*
  `engine/Dockerfile:47` imposta `ENV EULLM_AUDIT_DIR=/data/audit` e `docker-compose.yml`
  monta un volume su quel percorso, ma **la variabile non è letta da nessuna parte nel
  codice**: `AuditLogger::default_path()` (`audit/mod.rs:88-96`) usa esclusivamente
  `HOME`/`USERPROFILE`. In ogni deployment containerizzato il registro finisce nel layer
  effimero del container e si perde alla ricreazione, mentre il volume montato resta vuoto.
  Leggere la variabile come già si fa per `EULLM_MODELS_DIR` (`models/store.rs:57`) e
  rifiutare l'avvio se la directory non è scrivibile. Una riga per il fix, più il controllo
  all'avvio.

- [ ] **H0-C · L'adapter di identità deve arrivare nel GGUF finale** *(P0)*
  In `forge/eullm_forge/pipeline.py:158-176` lo stadio 4 assegna `adapter_path` ma non
  aggiorna `current_model_path` né fonde l'adapter nei pesi; lo stadio 5 esporta quindi il
  modello **pre-LoRA**. `eullm forge --identity "…"` completa senza errori e produce un
  GGUF privo dell'identità richiesta, dopo aver speso il tempo GPU dello stadio 4.
  Nota: `forge-research-roadmap.md` §F2 elenca l'identity LoRA come `[✅ implementato]` —
  il modulo lo è, l'orchestrazione che lo collega no. Fondere l'adapter
  (`merge_and_unload()`), salvare il checkpoint fuso, assegnarlo a `current_model_path`,
  e aggiungere un test che verifichi che il path esportato discende dallo stadio 4 quando
  `skip_identity` è falso.

- [ ] **H0-D · Riordinare la pipeline per il target GGUF** *(P0, insieme a H0-C)*
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

- [ ] **H0-E · NER attivo per default nell'anonimizzazione** *(P0)*
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

- [ ] **H0-F · Allineare il claim di anonimizzazione e rimuovere il riferimento alla fonte** *(P0)*
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

- [ ] **H1-A · Autenticazione opzionale a token con quote per chiave** *(P1)*
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

- [ ] **H1-B · `EULLM_ALLOWED_IPS` anche dall'ambiente di processo** *(P1)*
  `IpAllowlist::load_from_env_file` è invocata con il percorso letterale `".env"` relativo
  alla working directory (`api/mod.rs:436`) e la variabile non è mai letta dall'ambiente.
  `docker run -e EULLM_ALLOWED_IPS=…` e un `Environment=` in un'unità systemd non hanno
  alcun effetto, senza avviso; nessuna immagine del repository include un `.env`, quindi in
  container l'allowlist è sempre e solo loopback. Leggere la variabile dall'ambiente con
  precedenza sul file, e loggare all'avvio la sorgente effettiva della configurazione.

- [ ] **H1-C · Hardening del web tool** *(P1)*
  `tools::fetch_url` (`tools/mod.rs:77-96`) scarica la URL estratta dal messaggio utente
  senza validazione dell'host, senza limiti sul corpo della risposta
  (`resp.text()`, `:94`) e seguendo i redirect con la policy di default di reqwest.
  Aggiungere: risoluzione dell'host con verifica che ogni indirizzo risultante sia pubblico
  (escludere loopback, RFC1918, link-local, ULA IPv6), ripetizione del controllo su ogni hop
  di redirect o disabilitazione dei redirect, limite di byte leggendo a chunk invece che con
  `text()`, schema `https` per default con opt-in esplicito per `http`, e una allowlist di
  domini configurabile — che per il caso d'uso enterprise è la forma più difendibile della
  feature.

- [ ] **H1-D · Validare i nomi file restituiti dall'API HuggingFace** *(P1)*
  `list_hf_ggufs` raccoglie i `siblings[].rfilename` filtrandoli solo per il suffisso
  `.gguf` (`registry/mod.rs:543-551`), e `cmd_pull_hf` li usa direttamente come componente
  di percorso: `model_dir.join(&filename)` (`main.rs:920`). Un nome contenente separatori,
  `..` o un percorso assoluto non resta dentro la directory del modello. Rifiutare ogni
  `rfilename` che non sia un singolo componente sicuro, riusando la logica di
  `is_valid_model_slug` che esiste già in `hub/src/main.rs:356-365` invece di riscriverla.
  Applicare lo stesso controllo ai campi `gguf_file`/`mmproj_file` letti dal manifest
  (`models/store.rs:159, 193`).

- [ ] **H1-E · Restringere l'origine CORS e i percorsi modello accettati** *(P1)*
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

- [ ] **H1-F · Allineare l'Hub al perimetro dell'Engine** *(P1)*
  `hub/src/main.rs:65-69` ascolta su `0.0.0.0` senza allowlist, autenticazione, CORS né rate
  limit, a differenza dell'Engine. Riusare `ip_allowlist` (estraendolo in un modulo
  condiviso del workspace) e applicare H1-A quando disponibile. Correggere nello stesso
  passaggio la configurazione Docker: `hub/Dockerfile:38` dichiara `EXPOSE 3000` e il
  compose mappa `3000:3000`, ma il binario ascolta su 8080 in assenza di `EULLM_HUB_PORT`
  (`hub/src/main.rs:61-64`) — il servizio così com'è non è raggiungibile.

- [ ] **H1-G · Le card dell'Hub devono descrivere modelli esistenti** *(P1)*
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

- [ ] **H2-A · Dimensionare la KV cache con `attention.key_length`** *(P1)*
  Sia il sizer di `--fit` (`fit.rs:58-64`) sia la stima runtime (`scheduler.rs:1657`)
  calcolano `head_dim = n_embd / n_head`, ignorando le chiavi GGUF
  `<arch>.attention.key_length` / `.value_length` che molte architetture dichiarano
  esplicitamente. Su Qwen3-4B (n_embd 2560, n_head 32) il calcolo dà 80 contro un
  `head_dim` reale di 128: la KV è sottostimata del 37%, `--fit` offloada più layer di
  quanti entrino e il caricamento fallisce per OOM — su un modello presente nel catalogo.
  Leggere le due chiavi nel parser (che è già bounds-checked e facile da estendere) e usarle
  quando presenti, con l'attuale formula come fallback.

- [ ] **H2-B · Backpressure invece di scarto silenzioso dei token** *(P1)*
  `send_or_detect_disconnect` (`scheduler.rs:1734-1739`) tratta correttamente un canale
  pieno come "client lento, non disconnesso", ma **scarta il token** e prosegue: il client
  riceve testo con parti mancanti, senza alcun errore, e il conteggio finale non corrisponde
  a quanto consegnato. Applicare backpressure sulla sequenza (sospenderla per un'iterazione)
  oppure chiudere lo stream con un errore esplicito. Il canale ha capacità 256
  (`scheduler.rs:236`), quindi la condizione richiede un consumatore molto lento — ma è una
  perdita di dati silenziosa, non una degradazione.

- [ ] **H2-C · Trattare come errore il fallimento di `decode_batch.add`** *(P1)*
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

- [ ] **H2-G · Leggere l'header GGUF con `read_exact`** *(P2)*
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

---

## H3 — Igiene di processo e supply chain

**Gate di uscita:** la CI verifica ciò che la Definition of Done afferma; il mirror
garantisce ciò che dichiara; le dipendenze sono controllate per costruzione e non per
diligenza manuale.

- [ ] **H3-A · La CI deve compilare tutte le feature** *(P1)*
  La Definition of Done di `roadmap-engine-0.7-1.0.md` richiede che ogni voce «non rompa i
  feature flag CUDA, Metal, ROCm, Vulkan, multimodal», ma la CI esegue `cargo build` e
  `cargo test` solo con le feature di default (`.github/workflows/ci.yml:64, 67`): nessun
  percorso GPU e nessuna riga `#[cfg(feature = "multimodal")]` è mai compilata prima del
  push di un tag. È una Definition of Done non presidiata, in un repository che ha già
  imparato quanto costa scoprire un problema durante una release. Aggiungere un job
  `cargo check --features <x>` per ciascuna feature (economico: nessun link finale) e
  `cargo test --features multimodal` dove i test non richiedono GPU.

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

- [ ] **H3-D · Correggere i metadati di licenza del catalogo** *(P1)*
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

- [ ] **H3-K · Completare `SECURITY.md`** *(P3)*
  Il file è ancora il template GitHub non compilato: la tabella delle versioni supportate
  elenca `5.1.x` e `4.0.x`, che non esistono, e la sezione di segnalazione contiene le
  istruzioni segnaposto invece di un canale di contatto. Un progetto che pubblica binari e
  punta a un pubblico enterprise deve avere un canale di disclosure dichiarato e una policy
  sulle versioni supportate coerente con lo schema di release effettivo.

- [ ] **H3-L · Header di sicurezza sulla chat UI** *(P3)*
  `ui/mod.rs:58-79` serve gli asset con `Content-Type` e `Cache-Control` e nulla più. La
  disciplina di escaping in `app.js` è corretta — `renderContent` neutralizza l'HTML prima
  delle trasformazioni Markdown, incluse le sezioni di reasoning — ma un
  `Content-Security-Policy` restrittivo (la UI non carica nulla dall'esterno per progetto,
  quindi `default-src 'self'` è compatibile) più `X-Content-Type-Options: nosniff` sono una
  difesa in profondità a costo nullo che rende la proprietà "zero risorse esterne"
  verificabile dal browser invece che solo dichiarata.

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
