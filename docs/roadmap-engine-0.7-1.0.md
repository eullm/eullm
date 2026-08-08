# EULLM Engine — Roadmap tecnica 0.7 → 1.0

**Baseline:** Engine v0.6.15 · llama.cpp pinnato `9e3b928` · luglio 2026

Documento operativo: ogni voce ha una checkbox — sostituire `[ ]` con `[x]` (✅) al
completamento. Una voce è completata solo quando rispetta la Definition of Done
in fondo al documento.

---

## Principi vincolanti

- **Single binary**: il runtime resta distribuibile senza Python, Docker o servizi obbligatori.
- **llama.cpp come backend**: non duplicare kernel, quantizzazioni o primitive mantenute
  upstream. Quando libcommon ha già la funzionalità (grammar da JSON Schema, chat/tool
  template), esporla via wrapper C (`wrapper_common.cpp`, precedente: `llama_rs_fit_params`)
  invece di riscriverla in Rust.
- **Rust come control plane**: scheduler, API, audit, metriche, routing e lifecycle.
- **Portabilità**: nessun percorso ottimizzato solo-CUDA che degradi Metal, Vulkan, ROCm, CPU, ARM64.
- **Compatibilità**: non rompere i client Ollama/OpenAI esistenti.
- **Sovranità**: nessuna telemetria remota; metriche e audit locali per default.
- **Un cantiere per volta sullo scheduler**: mai due modifiche strutturali a
  `scheduler.rs` in parallelo. Ogni feature su branch dedicato, con benchmark
  prima/dopo sulla stessa macchina, stesso modello, stessi parametri.

---

## 0.7 — Misurare e non bloccare

**Gate di uscita:** TTFT, ITL, tempo di coda, prefill e decode misurati separatamente;
nessun blocco prolungato del decode durante prefill lunghi; riuso KV validato su hardware reale.

- [ ] **0.7-A · Validazione del KV prefix reuse sotto carico reale**
  Prima di ogni altro lavoro sullo scheduler. Conversazione CLI di 20 turni
  (verificare nei log `reused Y from cache` con Y crescente); 8 conversazioni
  concorrenti su `/api/generate` con prefissi crescenti indipendenti; cancellazioni
  (Ctrl-C, disconnessione client) a metà stream con turno successivo corretto;
  output byte-identico a seed fissato rispetto a v0.6.14 (reuse assente).

- [ ] **0.7-B · Metriche, osservabilità e benchmark** *(P0)*
  Endpoint `GET /health`, `GET /ready`, `GET /metrics` (Prometheus text format),
  `GET /api/stats` opzionale. Registry locale (counter/gauge/histogram, nessun invio
  remoto), aggiornato nel thread scheduler con costi da hot-loop trascurabili (atomics,
  niente lock nel decode loop). Metriche minime: richieste (total/running/waiting),
  tempi separati per coda/prefill/decode, TTFT, ITL, token prompt/generati, tok/s
  prefill e decode distinti, OOM, fallback KV, e per lo slot reuse: hit, miss,
  eviction, token riusati, token di prefill risparmiati. Label a cardinalità limitata
  (mai request_id, prompt, utente). Timestamp distinti: enqueue, admitted,
  prefill_start, decode_start, first_token, completed.
  Include il **fix della statistica CLI**: oggi il tok/s stampato a fine risposta
  include il tempo di prefill nel denominatore (il timer parte prima del prefill) —
  separare le due fasi anche nella riga `[model: N tokens, M prompt, X tok/s]`.
  Suite benchmark riproducibile (JSON/CSV): conversazione 20 turni, 8 conversazioni
  concorrenti, prompt RAG 4K/16K/32K/64K, mix corte/lunghe, coda satura,
  cancellazioni in ogni fase, cache slot piena, confronto KV F16/Q8_0/Q4_0,
  `batch_size=1` e `>1`.

- [ ] **0.7-C · Backpressure HTTP e deadline** *(P0 — prima parte del lifecycle)*
  Coda piena → HTTP 429 con `Retry-After` **prima** di aprire SSE/NDJSON (il
  `try_send` sullo scheduler fallisce già in modo sincrono: va solo intercettato
  prima dell'apertura dello stream); modello non disponibile → 503; validazione →
  400; prompt oltre il context → 413/422 con messaggio esplicito. Deadline
  opzionale per richiesta con `finish_reason` coerente e rilascio risorse.
  La cancellazione via disconnessione client (receiver drop) esiste già nel decode
  loop; l'endpoint `DELETE /api/requests/{id}` è rinviato a 0.9 (richiede il
  registry dei request_id, valore marginale finché il receiver-drop copre i casi reali).

- [ ] **0.7-D · Mixed chunked prefill** *(P0)*
  Oggi `prefill_sequence` decodifica tutti i chunk di un prompt lungo prima di
  restituire il controllo: le sequenze in streaming subiscono pause (head-of-line
  blocking). Rilevante solo con concorrenza (`eullm serve`); a `batch_size=1` il
  comportamento è invariato. Trasformare il prefill in stato incrementale
  (`SequencePhase::{Prefilling,Decoding,...}` + cursore) intercalato al decode:
  ogni iterazione decodifica un token per le sequenze attive, poi avanza il prefill
  di uno o più chunk entro un budget (`max_batched_tokens`, `prefill_chunk_tokens ≤ n_batch`).
  Policy iniziale decode-first, configurabile dopo benchmark.
  Invarianti da preservare (già presidiate dal reuse): logits richiesti solo
  sull'ultimo token del prompt completo; `n_past`/cursore/token registrati nello slot
  con un'unica fonte di verità; il campo `reused_prefix` del reuse è il punto di
  partenza del cursore. La cancellazione diventa verificabile anche tra i chunk di
  prefill (sinergia con 0.7-C). Testare prompt da 1, `n_batch` e `n_batch+1` token.
  Output identico a parità di seed rispetto al prefill monolitico.

- [x] **0.7-E · Auto-composizione `--fit` + `--n-cpu-moe`** *(implementato
  0.6.70-rc14)*
  Prima la scelta di N era manuale (trial-and-error documentato nel README).
  Implementato in `engine/src/fit.rs`: `parse_gguf_moe_layout`/
  `read_gguf_moe_layout` leggono la sezione tensor-info del GGUF (nome +
  offset per tensore — la dimensione reale viene dalla differenza tra
  offset consecutivi, non da un calcolo type/shape) e producono `MoeLayout`,
  la scomposizione per layer in byte expert vs non-expert. `compute_moe_fit`
  (puro, testato) calcola il minimo N di layer da spingere su CPU RAM
  (`--n-cpu-moe`) perché il resto entri in VRAM — evizione sempre di un
  prefisso contiguo `0..N` dal layer più basso, coerente con come
  `--n-cpu-moe` applica già il pattern per-layer. Se anche con tutti gli
  esperti su CPU RAM il resto non entra, ricade su uno split parziale a
  livello di layer intero calcolato sugli stessi byte non-expert (fino a
  interamente su CPU nel caso estremo) — riusa `compute_fit` esistente
  invece di duplicare la logica. Attivo solo con `--fit` e solo quando
  l'utente non ha già scelto lui `--cpu-moe`/`--n-cpu-moe` (rispetta
  l'intento esplicito). Non tocca `eullm serve`/`api::swap_model`, stessa
  scelta di scope già documentata per `--fit` in generale.

  Dati di calibrazione reali disponibili per un affinamento futuro (oggi il
  numero calcolato è il minimo che *entra*, non il più veloce): Qwen3.6-35B-A3B
  Q4_K_M su RTX 3060 12GB (26.5 tok/s blanket → 35.6 tok/s con N=24 + KV Q8_0)
  — restano validi come riferimento se in futuro si vorrà ottimizzare oltre
  al solo "deve partire".

  **Correzione rc15, trovata al primo test su hardware reale (7 agosto)**:
  proprio Qwen3.6-35B-A3B (UD-Q4_K_M, vocabolario da 248k token) faceva
  fallire `--fit` a monte — "could not parse layer count" → fallback a
  `--gpu-layers all` → OOM. Il parser dell'header leggeva i primi 8 MiB e
  scartava *tutto* se i metadati sforavano (gli array del tokenizer di quel
  modello da soli superano il budget), buttando via il `block_count` già
  letto 20 chiavi prima. Ora `parse_gguf_header` tollera il troncamento
  (restituisce il parziale, e si ferma appena ha tutti i campi voluti) e
  `read_gguf_moe_layout` — che la tabella tensori la trova solo *dopo*
  tutti i metadati, quindi il troncamento lì non è tollerabile — ritenta
  con budget crescenti (8→32→128 MiB). Spostata anche la decisione MoE
  *prima* del prompt continua/annulla di `run_fit`: risolve sempre in una
  configurazione caricabile, quindi non c'è niente da chiedere (prima il
  prompt citava uno split per layer interi che il passo MoE stava per
  sovrascrivere). Non è un problema solo MoE: anche Qwen3.6-27B dense
  (stesso vocabolario da 248k token, arch `qwen35` ibrida SSM) falliva
  identico su rc14 — stessa causa, stesso fix, il match per suffisso
  `.block_count` è agnostico all'architettura. **Validato su hardware
  reale (8 agosto)**: con rc15+ sia 35B-A3B (MoE, `GPU layers: all` +
  primi 17 layer di esperti su RAM, 38 tok/s su RTX 5070 Ti) sia 27B
  dense (split 51/64 a ctx 4096, 43/64 a ctx 16384) partono dal picker.

  **Estensione rc21, trovata dal vivo l'8 agosto**: il sizing valeva solo
  per il caricamento di lancio — uno swap dalla chat web (o via API)
  caricava il modello successivo con le impostazioni del modello di
  lancio: `run --fit` su 27B (43/64), switch al 35B MoE → niente offload
  esperti, split sbagliato, OOM. Ora `--fit`/`--fit-strict` sono in
  `RuntimeOpts` (esistono anche su `serve`) e `api::swap_model` esegue lo
  stesso sizing prima di ogni caricamento, dopo lo scarico del modello
  precedente (VRAM misurata reale), **senza mai chiedere conferma**
  (`run_fit_headless` — un daemon non ha nessuno alla tastiera;
  `--fit-strict` diventa un errore API). Il server eredita i flag
  originali dell'utente, mai i valori fittati sul modello di lancio.
  Validato su hardware reale con rc21: 27B via `--fit` → switch dalla
  chat al 35B MoE → caricamento riuscito con offload esperti, risposta a
  ~33 chunk/s.

---

## 0.8 — Contesto elastico e pipeline RAG completa

**Gate di uscita:** una richiesta singola usa il context pieno con gli altri slot liberi;
pipeline RAG (generazione + embedding + reranking) servita da un solo processo.

- [ ] **0.8-A · Scheduling a budget token + eviction slot cache** *(P0 — un solo branch, indivisibile)*
  Rimuovere lo split fisso `per_seq_ctx = ctx / max_batch_size`. **Vincolo di
  correttezza, non di performance**: oggi gli slot idle con KV residente non possono
  traboccare il pool condiviso proprio grazie allo split fisso (slot × per_seq_ctx =
  ctx totale); rimuovendolo, l'accounting a token (`used_active_kv + required ≤
  kv_budget`) e l'eviction LRU degli slot `IdleCached` (via `seq_rm`) devono
  arrivare **nello stesso branch**, altrimenti la cache degli slot può saturare le
  celle KV e far fallire richieste nuove. Ammissione:
  `required = prompt_tokens + reserved_output − reusable_prefix`; se manca spazio:
  evict LRU IdleCached → retry → accoda o rifiuta. Mai preemptare sequenze attive.
  La cache idle non deve mai impedire una richiesta nuova valida. Accounting a zero
  dopo unload. Fallback `batch_size=1` semplice e prevedibile. Metriche: token KV
  attivi, cached, riservati, evicted.

- [ ] **0.8-B · Embeddings e reranking in-process** *(P1)*
  `POST /api/embed`, `POST /v1/embeddings`, `POST /v1/rerank`. I binding espongono
  già `embeddings_seq_ith`/`embeddings_ith` e i pooling type (incluso `Rank` per il
  reranking): **secondo model slot in-process**, non worker/processi figli (i modelli
  embedding sono 100-600 MB e le chiamate sono stateless — un context dedicato con
  mutex è sufficiente; l'isolamento a processi è materia del gateway, 1.0+).
  Un modello embedding caricato all'avvio (flag dedicato), pooling coerente col
  modello, normalizzazione configurabile, batch multipli con ordine preservato,
  dimensione e modello dichiarati nella risposta. Le chiamate embedding non
  scaricano né bloccano il modello generativo.

- [ ] **0.8-C · Structured outputs completi** *(P1)*
  `response_format: json_schema` (formato OpenAI, `strict`) via
  `json-schema-to-grammar` **già presente in libcommon** — esporre con wrapper C,
  non riscrivere il compilatore di schemi in Rust. Estensioni:
  `grammar: {type: gbnf}` (il campo `grammar` in `GenerateRequest` esiste già) e
  `grammar: {type: choice}`. **Regex esclusa**: llama.cpp non la supporta e la
  conversione regex→grammar è un progetto a sé con valore di nicchia.
  Limiti su profondità/dimensione schema e lunghezza grammar; costrutti non
  supportati rifiutati esplicitamente, mai ignorati; errore chiaro se la grammar
  non si inizializza; `format=json` invariato; stesso risultato streaming e non;
  richieste concorrenti con grammar diverse senza contaminazione.

- [ ] **0.8-D · Tipi API per le route toccate**
  Tipizzare (via `api/types.rs`, `api/error.rs`) le sole route modificate da 0.8-B
  e 0.8-C, con golden test JSON/SSE/NDJSON prima della conversione. Nessuna
  riscrittura a tappeto delle route funzionanti: la migrazione completa procede
  opportunisticamente, route per route, quando una feature le tocca comunque.

---

## 0.9 — Agentic e verticale

**Gate di uscita:** tool calling validato su Qwen + una seconda famiglia;
modelli virtuali base+adapter funzionanti.

- [ ] **0.9-A · Tool calling OpenAI-compatibile** *(P1)*
  `tools`, `tool_choice` (none/auto/required/funzione specifica),
  `parallel_tool_calls`, messaggi `role=tool`, streaming dei delta.
  **Non implementare parser per-famiglia in Rust**: il llama.cpp pinnato ha già
  `common/chat.h` completo (template per famiglia, grammar constraining degli
  argomenti, parsing dell'output, JSON parziale per lo streaming) compilato in
  libcommon — esporre via wrapper C. Vantaggio strutturale: il supporto a nuove
  famiglie arriva con l'aggiornamento del pin upstream invece che con nuovo codice.
  Famiglie iniziali: Qwen, Mistral/Gemma. Modelli non supportati → errore esplicito,
  mai tool call inventate da parsing generico.

- [ ] **0.9-B · LoRA serving (adapter statici + modelli virtuali)** *(P1)*
  I binding espongono già `lora_adapter_init`/`lora_adapter_set`/`lora_adapter_remove`.
  **Vincolo di backend verificato**: l'adapter si applica per-context, non
  per-sequenza — adapter diversi nello stesso batch non sono supportati e il
  mixing per-richiesta su context condiviso serializzerebbe il continuous batching.
  Scope: adapter statico al lancio (`--adapter`) e "modelli virtuali"
  (`model: legal-it-studio-rossi` → base+adapter, risolti dal meccanismo di swap
  esistente). Chiave di validità dello slot/cache estesa: un prefisso KV è
  riusabile solo a parità di fingerprint modello **e** adapter (più chat-template,
  tokenizer, tipo KV). Audit: modello base, adapter, fingerprint, versione.
  Unload adapter senza scaricare il base. Tempistica allineata ai primi adapter
  prodotti da Forge.

- [ ] **0.9-C · Lifecycle completo delle richieste**
  `request_id` in risposta, registry delle richieste attive,
  `DELETE /api/requests/{id}`, priorità (Interactive/Normal/Batch) nella coda,
  graceful shutdown coordinato (stop ammissioni → completa o cancella secondo
  configurazione → rilascio modello) senza thread residui né VRAM occupata.
  Invariante cache: dopo cancellazione, uno slot resta `IdleCached` solo fino
  all'ultimo token per cui token registrati e KV sono verificabilmente allineati;
  nel dubbio, wipe completo (comportamento sicuro già in essere).

---

## 1.0+ — Solo su domanda dimostrata dai dati

- [ ] **1.0-A · Multimodale concorrente e multi-turno** *(P2)*
  Uscire dal percorso sequenziale forzato: content parts OpenAI (testo, immagine,
  audio) multi-turno, fingerprint SHA-256 dei media nella chiave di validità dello
  slot, encoding mtmd fuori dal decode loop, prefill media incrementale nello
  scheduler (dipende da 0.7-D). Nessuna condivisione KV tra media differenti.
  Percorso testuale puro senza regressioni a multimodal disabilitato.

- [ ] **1.0-B · Gateway e worker pool** *(P2 — attivare solo con requisito concreto)*
  Supervisor nello stesso eseguibile, worker come processi figli (generazione per
  modello, embedding, reranking, multimodale), routing per capability, restart con
  backoff, GPU assignment esplicito, readiness degradata, shutdown senza orfani.
  È l'item con il maggior rapporto complessità/valore: non avviarlo senza un
  deployment reale che lo richieda. La modalità single-process resta la primaria.

- [ ] **1.0-C · Speculative decoding N-gram** *(declassata da P1 — benchmark-gated)*
  Prompt-lookup/N-gram senza secondo modello, con auto-disattivazione sotto
  soglia di acceptance o alta concorrenza. Tre ragioni del declassamento:
  (1) sui carichi MoE con esperti su CPU il decode è compute-bound e la
  speculazione — che scambia compute per latenza — rende poco o nulla;
  (2) ristruttura il decode loop (token variabili per iterazione, rollback KV su
  rejection) sovrapponendosi ai cantieri 0.7-D/0.8-A; (3) senza le metriche di
  0.7-B non è dimostrabile che l'ITL sia il collo di bottiglia. Procedere solo se
  i dati raccolti lo giustificano; draft model solo dopo l'N-gram misurato.

- [ ] **1.0-D · Completamento tipi API e OpenAPI**
  Migrazione delle route residue da `serde_json::Value` a tipi stabili,
  `/v1/completions`, `/v1/responses`, `logprobs`, `stream_options.include_usage`,
  usage dettagliato, error object coerente, specifica OpenAPI quando i tipi sono
  stabili. Golden test per ogni route convertita; alias e campi Ollama preservati.

---

## Esclusioni permanenti (con motivazione verificata sul sorgente pinnato)

| Esclusa | Perché |
|---|---|
| APC globale stile vLLM (hash cross-conversation dei blocchi KV) | La KV cache di llama.cpp è una tabella piatta indicizzata per posizione e seq_id: nessuna primitiva a blocchi/hash su cui appoggiarsi. Andrebbe costruito un memory manager paginato sopra il backend — fuori scala e fuori missione. |
| PagedAttention / RadixAttention proprietari | Stessa ragione: duplicherebbero il backend invece di usarlo. |
| Prefill/decode disaggregati su nodi distinti | Non pertinente per appliance e GPU consumer. |
| Kernel CUDA proprietari | Comprometterebbero il modello multipiattaforma. |
| Tensor parallel proprietario | Prima esporre e validare le primitive multi-GPU già in llama.cpp. |
| Grammar da regex | Non supportata dal backend; conversione regex→GBNF è un progetto a sé con domanda di nicchia. Approssimabile con `choice`/GBNF. |
| session_id / seconda cache conversazionale | Lo slot reuse content-addressed (LCP a livello di token id) già in produzione copre il caso senza parametri nuovi. |

---

## Nota sui costi di performance

- **Metriche**: costo ~zero con atomics nel thread scheduler; vietati lock nel decode loop.
- **Chunked prefill**: neutro-positivo sulla latenza percepita; costo throughput marginale, policy configurabile.
- **Budget token**: puro control plane, trascurabile.
- **Structured outputs**: costo per-token del grammar sampling già presente oggi con `format=json`; conversione schema una-tantum per richiesta.
- **Embeddings in-process**: costo = VRAM/RAM del secondo modello (piccolo); elimina un runtime esterno.
- **Speculative**: unica voce che può regredire le performance (bassa acceptance, alta concorrenza) — da qui il gate sui benchmark.
- **Gateway**: overhead di processo e complessità operativa — da qui il rinvio a domanda reale.

---

## Definition of Done (per ogni voce)

- Compila in CPU e non rompe i feature flag CUDA, Metal, ROCm, Vulkan, multimodal.
- `cargo fmt --check`, `cargo clippy` (`-D warnings`) e `cargo test` puliti.
- CI esistente non semplificata; caching non rimosso.
- Percorso sequenziale e continuous batching entrambi funzionanti; fallback testato.
- Route Ollama e OpenAI senza regressioni; streaming e non-streaming testati.
- Test: positivo, di errore e di concorrenza (dove applicabile); golden test per le API.
- Benchmark prima/dopo su stessa macchina, stesso modello, stessi parametri.
- Metriche o log dimostrano che il percorso nuovo è realmente esercitato.
- Nessun dato personale o contenuto integrale in metriche e label.
- README, `--help` e release notes aggiornati; feature sperimentali disattivabili.
