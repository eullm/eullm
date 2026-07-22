# Dossier conformità — AI Act & GDPR per l'ecosistema EULLM + I3K RAG Enterprise

> **Scopo**: materiale di preparazione per la call con **Etel Friedmann (Lunar Ventures)**.
> Fotografa la **situazione attuale** (cosa fa davvero il codice oggi) e la **situazione
> futura** (cosa impone la normativa, cosa dovremo costruire), con il posizionamento:
> **noi forniamo gli strumenti per mettere in regola gli utilizzatori, non ci assumiamo
> la loro responsabilità**.
> Aggiornato al **22 luglio 2026**. Le fonti sono elencate in fondo (§12).

---

## 0. Executive summary — le 8 cose da sapere a memoria per domani

1. **L'AI Act NON entra "al 100%" il 2 agosto 2026.** Il *Digital Omnibus on AI* (adottato
   dal Parlamento UE il 16 giugno 2026, dal Consiglio il 29 giugno 2026, in pubblicazione
   sulla Gazzetta Ufficiale) **ha rinviato il cuore degli obblighi sull'alto rischio**:
   Annex III stand-alone dal 2 ago 2026 → **2 dicembre 2027**; Annex I integrato → **2 ago 2028**.
2. **Cosa è già in vigore o scatta comunque a breve:** pratiche vietate (feb 2025), **obblighi
   GPAI** + governance + sanzioni (ago 2025), **trasparenza Art. 50** (2 ago 2026, marcatura
   contenuti sintetici con proroga al 2 dic 2026). Questi *non* sono stati rinviati.
3. **Dove si colloca EULLM nella catena del valore:** Forge/Hub ci rendono **provider (o
   downstream provider) di un modello GPAI**; l'Engine è un **abilitatore** (chi lo usa è
   *deployer*). Le responsabilità pesanti (alto rischio) sono **dell'utilizzatore/deployer**,
   non nostre. L'Omnibus ha **rafforzato l'Art. 25**: il fornitore a monte deve passare a valle
   documentazione, limiti noti e accesso per i test → **è esattamente la funzione della scheda
   di conformità**.
4. **Open source riduce ma non azzera gli obblighi:** anche per un modello aperto restano
   **obbligatori** la *policy sul copyright* (Art. 53(1)(c)) e il *riassunto pubblico dei dati
   di addestramento* (Art. 53(1)(d)). I nostri 7B sono ben sotto la soglia di rischio sistemico
   (10²⁵ FLOP): siamo nella corsia "GPAI ordinario".
5. **Alto rischio = dipende dall'uso, non dal dominio.** `legal-it-7b` non è "ad alto rischio"
   perché parla di diritto; lo diventa se un deployer lo usa per assistere un giudice (Annex III
   §8), per il credit scoring (§5), in ambito medico-dispositivo (Annex I), ecc. La nostra leva:
   **model card che delimitano l'uso previsto**.
6. **Cosa fa DAVVERO il codice oggi:** funzionano quattro cose — (a) audit log locale *solo
   metadati*, (b) zero telemetria, (c) IP allowlist in ingresso, (d) un **anonimizzatore PII
   reale in Forge**. Le "schede di conformità" del Hub **esistono ma sono statiche/hardcoded,
   identiche per ogni modello** e affermano cose che non reggerebbero a una verifica. Due frasi
   del README sono sovradimensionate ("audit di ogni request/response", "tutti i dati restano
   nei confini UE"). **Questo è il gap → è la roadmap → è la ragione del round.**
7. **rag-enterprise.com è un nostro prodotto (I3K RAG Enterprise, AGPL-3.0)** e usa già EuLLM
   come uno dei modelli. Storia da raccontare: **stack sovrano a due livelli** — EULLM (modelli
   + inferenza) e RAG Enterprise (applicazione). Le schede di conformità del Hub alimentano il
   "high-risk documentation pack" di RAG Enterprise. Integrazione verticale = vantaggio.
8. **GDPR + on-prem = vantaggio strutturale.** L'inferenza locale (nessun dato che esce)
   toglie di mezzo il Capo V (trasferimenti, rischio "Schrems III") e ci rende **fornitore di
   software, non responsabile del trattamento** del cliente. È un argomento durevole che un
   concorrente su cloud USA non può replicare.

**La frase-chiave per Etel:** *«L'AI Act non punisce chi costruisce lo strumento, ma chi lo
mette sul mercato e chi lo usa senza documentazione. Noi vendiamo esattamente quella
documentazione — resa automatica, verificabile e sovrana — così i nostri utilizzatori possono
mettersi in regola. Non ci assumiamo la loro responsabilità: gli diamo gli attrezzi per
assumersela in modo difendibile.»*

---

## 1. La correzione che cambia il pitch: la timeline reale dell'AI Act

La premessa diffusa ("l'AI Act entra in vigore al 100% ai primi di agosto") **era vera fino a
novembre 2025 ed è stata superata**. Portarla in una call con un fondo significherebbe partire
con un dato sbagliato; portarla *corretta* ci fa sembrare i più aggiornati nella stanza.

### 1.1 Il Digital Omnibus on AI (iter concluso)
- **19 nov 2025** — la Commissione propone il pacchetto "Digital Omnibus" (semplificazione).
- **16 giu 2026** — il **Parlamento UE** lo adotta (423 favorevoli / 57 contrari / 174 astenuti).
- **29 giu 2026** — il **Consiglio** dà il via libera finale.
- **luglio 2026** — pubblicazione in Gazzetta Ufficiale UE; entra in vigore il 3° giorno dopo la
  pubblicazione (attesa *prima* del 2 agosto 2026).
- Il meccanismo "a scadenza condizionata" (agganciato alla disponibilità degli standard),
  inizialmente proposto, è stato **sostituito da date fisse**.

> **Cautela onesta da dire a Etel:** il testo definitivo in Gazzetta è di pochissime settimane
> fa; le date qui sotto convergono in tutte le analisi (Consiglio, Freshfields, Gibson Dunn,
> White & Case), ma "la GU è l'autorità ultima" — un modo elegante per mostrare rigore.

### 1.2 Timeline effettiva (vecchie vs nuove date)

| Data | Cosa scatta | Stato |
|---|---|---|
| 1 ago 2024 | Entrata in vigore del Regolamento (UE) 2024/1689 | fatto |
| **2 feb 2025** | **Pratiche vietate (Art. 5)** + alfabetizzazione IA (Art. 4) | **in vigore** (invariato) |
| **2 ago 2025** | **Obblighi GPAI (Art. 51–56)** + governance + **sanzioni (Art. 99–101)** | **in vigore** (invariato) |
| **2 ago 2026** | Applicazione generale del resto del Regolamento + **trasparenza Art. 50** | **scatta** (ma NON è "tutto") |
| ~~2 ago 2026~~ → **2 dic 2026** | Marcatura machine-readable dei contenuti sintetici (Art. 50(2)) — proroga per i sistemi legacy; nuovi divieti Art. 5 (deepfake intimi/"nudifiers", CSAM) | rinviato dall'Omnibus |
| ~~2 ago 2026~~ → **2 dic 2027** | **Alto rischio Annex III stand-alone** (Art. 8–15, 16 provider; Art. 26 deployer; valutazione di conformità; registrazione in banca dati UE) | **rinviato** (il grande) |
| **2 ago 2027** | Modelli **GPAI legacy** (immessi prima del 2 ago 2025) devono essere conformi (Art. 111(3)) | invariato |
| ~~2 ago 2027~~ → **2 ago 2028** | **Alto rischio Annex I integrato** (IA in prodotti regolati: dispositivi medici, macchine…) | **rinviato** |

**Cosa "morde" davvero il 2 agosto 2026:** disposizioni generali + **Art. 50 (trasparenza)** —
disclosure del chatbot, etichettatura deepfake. La macchina pesante del Capo III (alto rischio)
**non** si applica fino a dic 2027 / ago 2028.

### 1.3 Perché per noi è un'opportunità, non un problema
- **~16 mesi in più** per arrivare pronti sull'alto rischio proprio nelle categorie che ci
  interessano (legal, medical, finance = Annex III §5/§8 e Annex I medicale).
- Ci posiziona come chi **arriva in anticipo con gli strumenti**, quando il mercato è ancora
  in fase di preparazione: *timing perfetto per un investimento*.
- Il 2 ago 2026 resta comunque una **scadenza attiva** (Art. 50): c'è un deliverable concreto e
  a breve termine (disclosure "stai parlando con un'IA") che possiamo spuntare subito — segnale
  di execution per il fondo.

---

## 2. Visione e missione — come raccontarle a Etel

**Visione.** L'Europa avrà bisogno di uno *stack di IA sovrano e conforme by-design*: modelli
permissivi, verticalizzati per dominio e lingua, che girano **dentro il perimetro del cliente**
senché un byte esca verso cloud extra-UE, con la **documentazione di conformità inclusa**.

**Missione.** Rendere la conformità (AI Act + GDPR) un **requisito di ingegneria di prima
classe**, non una checklist di marketing: strumenti che *generano* la documentazione, *tracciano*
l'inferenza e *delimitano* l'uso — così che gli utilizzatori possano mettersi in regola in modo
difendibile.

**Lo stack a due livelli (il "perché adesso" per il fondo):**

| Livello | Prodotto | Ruolo AI Act | Cosa vende |
|---|---|---|---|
| Modelli + inferenza | **EULLM** (Engine, Forge, Hub) | *Provider/downstream provider* di modelli GPAI aperti + abilitatore di inferenza | Modelli verticalizzati sovrani + **schede di conformità** + audit trail |
| Applicazione | **I3K RAG Enterprise** (rag-enterprise.com) | Sistema che il cliente usa (*deployer*) — potenzialmente alto rischio nel suo contesto | RAG on-prem AGPL + "high-risk documentation pack" |

Il **valore composto**: le schede di conformità prodotte dal Hub sono l'**input** al pacchetto
documentale di RAG Enterprise, che è a sua volta l'input alla DPIA/valutazione del deployer.
È una **catena di conformità verticalmente integrata** — difficile da replicare, e allineata a
un mercato (PA, legale, sanità, finanza UE) che *deve* comprare sovrano.

**Il posizionamento sulla responsabilità (da ripetere spesso):** non vendiamo "conformità
garantita" — la conformità è una proprietà dell'**intero sistema e della sua governance**, non
di un binario. Vendiamo gli **strumenti** che permettono all'utilizzatore di dimostrare la
propria. Questo ci tiene fuori dalla catena della responsabilità e *dentro* al mercato degli
enabler. (Il README lo dice già: *"We make no claim that a binary makes a system AI Act
compliant"* — teniamolo e valorizziamolo.)

---

## 3. L'ecosistema EULLM OGGI — cosa fa davvero il codice

> Sezione **grounded sul codice reale** (non sul README). Serve a evitare figuracce: se Etel
> fa due domande tecniche, dobbiamo distinguere ciò che *funziona* da ciò che è *roadmap*.

### 3.1 Engine (Rust) — cosa gira davvero
**Funziona (in codice):**
- **Audit trail locale** (`engine/src/audit/mod.rs`): scrive un JSONL append-only in
  `~/.eullm/audit/audit.jsonl`, **attivo di default**, cablato su **tutte** le API (Ollama
  `/api/generate`, `/api/chat`, OpenAI `/v1/chat/completions`), sia streaming che non.
  Campi registrati: `id` (UUID), `timestamp`, `model`, `request_type`, `input_tokens`,
  `output_tokens`, `duration_ms`. Sanitizzazione anti log-injection.
- **Zero telemetria** (vero): nessuna analitica/crash-report; UI embedded self-contained
  (nessun CDN/font/tracker) — *"sovereign by default"*.
- **IP allowlist in ingresso** (`engine/src/api/ip_allowlist.rs`, default solo loopback) —
  controllo d'accesso.
- **Verifica SHA-256** dei pesi scaricati + `manifest.json` locale per-modello (provenienza minima).

**Gap onesti (da presentare come roadmap, non da nascondere):**
- L'audit registra **solo metadati**: *non* il testo di prompt/risposta e *non* l'utente finale
  (il campo `user_id` esiste ma non viene mai popolato). → Il README (`README.md:118`) afferma
  *"audit trail of every request/response"*: **sovradimensionato**, va corretto o va implementato.
- **Nessuna residenza dati UE forzata**: i download dei modelli passano da `huggingface.co`
  (CDN USA), in contraddizione con la scheda del Hub che afferma *"All data stays within EU
  borders"*.
- **Nessuna funzione di trasparenza Art. 50** (disclosure "sei un'IA", marcatura contenuti
  sintetici/watermark): assente del tutto. → È il deliverable a breve per il 2 ago 2026.

### 3.2 Forge (Python) — pipeline e provenienza
**Funziona:**
- Pipeline `pruning → distillation → quantization → identity LoRA → GGUF export`, con i profili
  di dominio `legal_it / medical_de / finance_fr` (config di compressione: base Qwen3-14B →
  prune 0.5 mlp-first → distill → AWQ 4-bit → LoRA identità → GGUF q4_k_m).
- **Anonimizzatore PII reale e sostanziale** (`forge/eullm_forge/datasets/anonymize.py`):
  redige codice fiscale, P.IVA, IBAN, email, telefoni, clausole di nascita, indirizzi da
  sentenze di Cassazione *prima* dell'addestramento; regex sempre attive + NER opzionale;
  statistiche di redazione per record; irreversibile. La CLI rifiuta di pubblicare dati grezzi
  di Cassazione per motivi GDPR. **Questo è un controllo GDPR vero e vendibile.**

**Gap:**
- **Nessuna generazione di scheda/model card**: la pipeline emette *solo pesi/GGUF*. Base model,
  dataset, iperparametri restano nel profilo YAML di input e nei log stdout — **non** vengono
  persistiti come artefatto di provenienza accanto al modello. → Da costruire (è il cuore della §7).
- Il fine-tuning "identità" fa *affermare* al modello di essere "GDPR compliant": è un claim
  nella risposta, **non** un controllo tecnico. Da non spacciare per conformità.

### 3.3 Hub (Rust) — registry e schede
**Funziona:** endpoint `/{name}/card` (model card) e `/{name}/compliance` (scheda di conformità),
`/v1/models`, `/{name}/download`, con hardening anti path-traversal.

**Gap critico (da correggere per primo):**
- Le schede di conformità **sono statiche e hardcoded**, *identiche byte-per-byte per ogni
  modello* (il nome è l'unico campo interpolato). Affermano tra l'altro: `risk_classification =
  "GPAI"`, `systemic_risk = false`, `gdpr_compliant = true`, `personal_data = "No personal data
  in training set"`, `right_to_erasure = "Not applicable"`, `data_residency = "All data stays
  within EU borders"`. **Diverse di queste affermazioni non reggerebbero a una due-diligence**
  (es. "nessun dato personale" per un modello addestrato su Cassazione; `gdpr_compliant=true`
  come asserzione secca; `systemic_risk=false` hardcoded).
- Il catalogo è uno **stub hardcoded**; i download in gran parte danno 404. **I modelli demo non
  esistono ancora** (lo ammette il README). ⇒ Coerente con il fatto che **le schede si
  riferiscono ai modelli verticalizzati** (futuri), **non** a quelli scaricabili oggi.

### 3.4 Quadro sintetico "implementato / parziale / assente"

| Capacità | Stato oggi | Dove |
|---|---|---|
| Audit log locale (metadati), on-by-default, su tutte le API | **Implementato** | `engine/src/audit/mod.rs` |
| Audit con **contenuto** request/response | Assente (README lo claima) | — |
| Audit con identità utente ("chi") | Assente (`user_id` mai valorizzato) | — |
| Export/report/query dell'audit | Assente | — |
| Zero telemetria | **Implementato** (come assenza) | `engine/src/ui/mod.rs` |
| IP allowlist in ingresso | **Implementato** | `engine/src/api/ip_allowlist.rs` |
| Verifica SHA-256 pesi | **Implementato** | (commit recente) |
| Residenza dati UE **forzata** | Assente/aspirazionale (download da HF) | `engine/src/registry/mod.rs` |
| Trasparenza Art. 50 (disclosure IA / watermark) | Assente | — |
| Anonimizzazione PII dati di training | **Implementato** | `forge/.../datasets/anonymize.py` |
| Generazione model card / scheda in Forge | Assente | — |
| Artefatto di provenienza col modello | Parziale (input YAML/log, non persistito) | — |
| Scheda di conformità del Hub | **Implementata ma statica/hardcoded** | `hub/src/main.rs` |
| Catalogo Hub reale (DB) + download modelli | Stub/doc-only; modelli demo inesistenti | `hub/src/main.rs` |

---

## 4. Come l'AI Act impatta ogni componente

### 4.1 Ruoli e catena del valore (il punto giuridico centrale)
Definizioni (Art. 3): **provider** = chi sviluppa e immette sul mercato/mette in servizio
**a proprio nome o marchio, anche a titolo gratuito** (3(3)); **deployer** = chi lo usa
(3(4)); **downstream provider** = chi integra un modello IA in un sistema (3(68));
**substantial modification** = modifica non prevista nella valutazione di conformità iniziale
(3(23)).

- **Forge/Hub** — comprimiamo/fine-tuniamo un modello aperto e lo **ridistribuiamo sotto il
  marchio `eullm/…`** ⇒ siamo (quasi certamente) **provider** del modello/sistema risultante.
  Se ereditiamo i pieni obblighi da *provider di modello GPAI* dipende dal **test dell'1/3 del
  compute** (Linee guida GPAI della Commissione, lug 2025 + Considerando 109): se una modifica
  usa **≥ 1/3 del compute** originale di addestramento, si *presume* di essere diventati
  provider GPAI. Pruning+distillation di un 14B→7B è il nostro caso più vicino a quella soglia;
  un LoRA di identità in genere no. **Prudenza**: budgetiamo come se fossimo provider GPAI
  (riassunto training + policy copyright + doc Annex XI/XII). Nota: gli obblighi del downstream
  sono **limitati alla modifica**, non all'intero modello a monte.
- **Engine** — far girare/servire un modello è **deploying**, non "providing", a meno che non ci
  si metta un marchio o si modifichi sostanzialmente. Distribuire un *motore* di inferenza
  (come Ollama/llama.cpp) non è di per sé "fornire un modello". ⇒ Engine = **abilitatore**.
- **L'Omnibus ha rafforzato l'Art. 25**: il provider a monte deve fornire al downstream (a) doc
  tecnica sufficiente a valutare la conformità all'Art. 16, (b) info su limiti noti e modalità di
  guasto, (c) accesso tecnico mirato per test/validazione; "AI model" è ora esplicito nell'obbligo
  di accordo scritto (Art. 25(4)); le violazioni salgono alla fascia **3% / €15M**. → **La scheda
  di conformità È l'artefatto di questo passaggio Art. 25.**

### 4.2 Esenzioni open source — e i loro limiti
Due meccanismi distinti:
- **Sistemi** (Art. 2(12)): il Regolamento **non si applica** ai sistemi IA rilasciati con
  licenza libera/open source, **salvo** che siano alto rischio, vietati (Art. 5) o soggetti a
  trasparenza (Art. 50). Esenzione **stretta**: appena l'uso è ad alto rischio o rilevante per
  la trasparenza, svanisce.
- **Modelli GPAI** (Art. 53(2)): i modelli genuinamente aperti (pesi, architettura e info d'uso
  pubblici) sono esenti **solo** dagli obblighi di *doc tecnica* (53(1)(a)) e *info al
  downstream* (53(1)(b)). **Restano obbligatori** la **policy sul copyright** (53(1)(c), incl.
  opt-out TDM della Dir. 2019/790) e il **riassunto pubblico dei dati di training** (53(1)(d)).
  L'esenzione **non** vale per i modelli a **rischio sistemico** (≥10²⁵ FLOP).

**Per noi:** open source **riduce** ma **non elimina**. Per ogni modello del Hub va pubblicato:
riassunto dei contenuti di training (template ufficiale AI Office, del 24 lug 2025) + policy
copyright/TDM. Siamo lontanissimo dai 10²⁵ FLOP ⇒ nessun obbligo da rischio sistemico.

### 4.3 Alto rischio (Annex III) — dipende dall'uso
Un modello "legale/medico/finanziario" **non è automaticamente** alto rischio. Lo diventa se
l'**uso previsto** rientra in Annex III (Art. 6(2)), con un **filtro** (Art. 6(3)): non è alto
rischio se non pone un rischio significativo e svolge solo compiti procedurali ristretti,
migliorativi o preparatori — **ma qualunque *profilazione* di persone fisiche è *sempre* alto
rischio**.

Categorie Annex III rilevanti per i nostri verticali: **§3 istruzione**, **§4 lavoro/HR**,
**§5 servizi essenziali** (incl. **creditworthiness/credit scoring**, pricing assicurazioni
vita/salute), **§8 amministrazione della giustizia**, **§1 biometria**. Il **medicale** può
inoltre ricadere in **Annex I / normativa dispositivi medici (MDR)** — è il verticale più
rischioso.

Obblighi del **provider** di sistema ad alto rischio (Art. 16 → 8–15): gestione del rischio,
data governance, doc tecnica (Annex IV), logging, trasparenza verso il deployer, sorveglianza
umana, accuratezza/robustezza/cybersecurity; + sistema di qualità (Art. 17), valutazione di
conformità (Art. 43), dichiarazione UE (Art. 47), marcatura CE (Art. 48), **registrazione in
banca dati UE** (Art. 49). Obblighi del **deployer** (Art. 26): uso secondo istruzioni,
sorveglianza umana competente, monitoraggio, **log ≥6 mesi**, informare i lavoratori,
DPIA/valutazione impatto sui diritti fondamentali, informare gli interessati.

**Leva EULLM:** model card che **delimitano l'uso previsto** e segnalano "non destinato a
\[uso ad alto rischio] senza valutazione di conformità del deployer". Sposta la responsabilità
sull'uso, dove deve stare.

### 4.4 Trasparenza (Art. 50) — il deliverable a breve
- **50(1)** i sistemi che interagiscono con persone devono dichiarare di essere IA → **chatbot
  dell'Engine**.
- **50(2)** i contenuti sintetici (testo/audio/immagini/video) vanno marcati in formato
  machine-readable (watermark/provenance) → **dal 2 dic 2026** (proroga per i sistemi legacy).
- **50(4)** i deployer etichettano i deepfake e i testi IA pubblicati su temi di interesse
  pubblico.

⇒ **To-do concreto entro il 2 ago 2026**: aggiungere all'Engine la disclosure "stai interagendo
con un sistema IA" (banner/response header/field OpenAI). Piccolo, spuntabile, dimostra execution.

### 4.5 Sanzioni (Art. 99 / 101)
Pratiche vietate: fino a **€35M o 7%** del fatturato mondiale. Gran parte degli obblighi (incl.
Art. 50 e ora Art. 25(2)/(4)): **€15M o 3%**. Info errate alle autorità: **€7.5M o 1%**.
Provider GPAI (Art. 101): **€15M o 3%**. **Start-up/PMI**: si applica il *minore* tra importo e
percentuale (rilevante per noi).

---

## 5. GDPR e le altre norme ("varie")

### 5.1 GDPR applicato a LLM e RAG
- **Base giuridica (Art. 6):** per training/fine-tuning su dati personali la base realistica è il
  **legittimo interesse (Art. 6(1)(f))**, con **LIA** documentata in 3 passi (interesse legittimo
  specifico e attuale; necessità; bilanciamento). Avallato *condizionatamente* da EDPB
  Opinion 28/2024 e CNIL (giu 2025). All'**inferenza**, ogni query con dati personali è un
  trattamento a sé: in un prodotto enterprise la base la mette il **cliente (titolare)**.
- **Categorie particolari (Art. 9)** — critico per `medical-de` e `legal-it`: salute, dati
  biometrici, ecc. sono **vietati** salvo eccezione (consenso esplicito, interesse pubblico
  rilevante con base normativa, ricerca con Art. 89…). Il legittimo interesse **non** basta:
  serve una **doppia base** (Art. 6 *e* Art. 9). L'anonimizzatore di Forge è la risposta tecnica
  giusta, ma **i dati pseudonimizzati restano dati personali**.
- **Titolare vs responsabile (Art. 4/28) — il vantaggio on-prem:** in un deployment self-hosted
  il **cliente è titolare e responsabile di sé stesso**; **I3K/EULLM è solo fornitore di
  software**, non responsabile del trattamento dei dati a runtime. Niente DPA per il flusso
  runtime, niente catena di sub-responsabili, **niente trasferimento internazionale**. È
  l'argomento di data-protection più forte dell'architettura. (Cautela: restano flussi
  accessori — telemetria, accesso di supporto, server di update/registry — da tenere UE e
  documentati.)
- **Residenza e trasferimenti (Capo V):** un'architettura **UE-resident senza egress** toglie di
  mezzo l'intero Capo V. L'EU–US Data Privacy Framework è **valido ma contestato** (Tribunale UE
  l'ha confermato il 3 set 2025; ricorso alla CGUE; nuova sfida dopo la sentenza USA del 29 giu
  2026 sull'indipendenza della FTC) ⇒ **rischio "Schrems III"** che rafforza il pitch on-prem.
- **DPIA (Art. 35):** per deployment LLM/RAG su dati personali **assumere che sia obbligatoria**
  (scala, categorie particolari, profilazione, tecnologia nuova). L'AI Act dice ai deployer di
  **riusare** il lavoro DPIA con la doc del provider (Art. 13): la nostra scheda alimenta la DPIA.
- **Decisioni automatizzate (Art. 22) + sentenza SCHUFA (C-634/21):** produrre un *punteggio*
  (es. credit score) è già "decisione automatizzata" se un terzo vi si basa pesantemente. ⇒ se
  un modello EULLM/RAG *emette* una decisione/punteggio in finanza/legale/medicina, l'Art. 22
  morde. **Mitigazione: human-in-the-loop by design** (allineato all'Art. 14 AI Act).
- **Diritti e il problema della cancellazione (Art. 15–17):**
  - **Vector store del RAG:** accesso/rettifica/**cancellazione sono trattabili** — documenti ed
    embedding sono indirizzabili e cancellabili (RAG Enterprise già offre "per-document/per-user
    deletion"). **Vantaggio del RAG sulla memoria parametrica.**
  - **Pesi del modello:** se il dato personale era nel training, può essere "cotto" nei pesi e
    non c'è cancellazione pulita per-record. **EDPB Opinion 28/2024**: un modello addestrato su
    dati personali **non è automaticamente anonimo** (test caso per caso, soglia alta:
    probabilità di estrazione "insignificante"). **Stance strategica:** preferire **RAG al
    fine-tuning** per i dati personali/sensibili, minimizzare PII nei corpora, documentare
    unlearning/rollback.
- **EDPB Opinion 28/2024 (17 dic 2024)** è il documento-ancora: (1) anonimato = soglia alta,
  caso per caso; (2) legittimo interesse ammesso ma solo dopo il test in 3 passi + mitigazioni
  (opt-out, de-identificazione, trasparenza, filtri in output); (3) se il modello è stato
  costruito illecitamente, il deployer a valle deve fare **due diligence**; (4) scraping
  indiscriminato difficile da giustificare. Follow-up: report EDPB "AI Privacy Risks &
  Mitigations — LLMs" (10 apr 2025, *Support Pool of Experts*, non posizione ufficiale); CNIL
  fiches pratiques (feb e lug 2025).

### 5.2 Le altre norme (solo le parti che mordono)
- **Product Liability Directive rivista — Dir. (UE) 2024/2853:** software e IA sono
  esplicitamente **"prodotti"** in **responsabilità oggettiva** (senza colpa); recepimento entro
  **9 dic 2026**, applicazione ai prodotti immessi dopo tale data. La **FOSS fornita fuori da
  attività commerciale è esclusa**; il **software commerciale è pienamente incluso**. ⇒ poiché
  I3K **commercializza** (licenza perpetua + supporto), il regime PLD **si applica** alla
  fornitura commerciale. Non si può escludere per contratto. Un difetto può derivare da
  **aggiornamenti di sicurezza mancanti**.
- **Cyber Resilience Act — Reg. (UE) 2024/2847:** il software IA è **in scope**. Obblighi di
  segnalazione vulnerabilità dall'**11 set 2026**, obblighi pieni dall'**11 dic 2027**. La FOSS
  **genuinamente non-commerciale** è fuori; gli "steward" open source hanno un regime più
  leggero; la **fornitura commerciale = obblighi pieni di *manufacturer*** (secure-by-design,
  gestione vulnerabilità, **SBOM**, aggiornamenti, marcatura CE). ⇒ per il prodotto commerciale
  I3K: **pianificare compliance CRA da manufacturer**; tenere pulito il confine dell'edizione
  community non-commerciale.
- **DORA — Reg. (UE) 2022/2554:** applicabile dal **17 gen 2025**; vincola le entità finanziarie
  e i loro **fornitori ICT terzi**. Se forniamo servizi ICT continuativi a un cliente
  finanziario, possiamo essere **fornitore ICT terzo** soggetto a requisiti contrattuali DORA.
  ⇒ Precisare "data center DORA-compliant" come proprietà del *cliente/host*, non del software.
- **NIS2 — Dir. (UE) 2022/2555:** recepimento a macchia di leopardo (alcuni Stati ancora
  indietro a metà 2026). Vincola le entità essenziali/importanti = **molti nostri clienti**, che
  chiederanno garanzie di sicurezza della supply chain → trasformarlo in requisito di vendita.
- **Data Act (Reg. 2023/2854, dal 12 set 2025)** e **Data Governance Act (Reg. 2022/868, dal set
  2023):** impatto marginale per un prodotto self-hosted/portabile; rileverebbero solo se il Hub
  diventasse un *servizio di intermediazione dati* o se offrissimo una versione *hosted* (regole
  di cloud-switching/portabilità).
- **AI Liability Directive: RITIRATA** (ritiro formale, avviso in GU ottobre 2025). La
  responsabilità da IA passa ora per la **PLD rivista** + diritto nazionale.

**Il crinale che conta per I3K:** la linea **commerciale vs non-commerciale** della OSS
determina l'esposizione a **CRA e PLD**. Va gestita a livello di termini di licenza/supporto.
(*Nota: la "Data Omnibus" lato GDPR — che cambierebbe la definizione di dato personale e
aggiungerebbe basi per il training IA — è **solo una proposta**, non legge: non costruirci sopra.*)

---

## 6. rag-enterprise.com (I3K RAG Enterprise) nella storia

**Cos'è (dal sito live):** piattaforma **RAG open-source (AGPL-3.0), self-hosted**, posizionata
come *"la piattaforma RAG open-source per organizzazioni che non possono mandare i dati ai cloud
americani"*. Deploy one-command su Ubuntu, **air-gapped**. Stack: Qdrant (vector store), LLM
**EuLLM**, Mistral 7B, Qwen3-14B-q4; embedding BAAI/bge-m3 (29 lingue); ingestion Apache Tika +
Tesseract OCR. Verticali: legale (collezioni per pratica, RBAC avvocato-cliente, provenienza
citabile, OCR on-prem), **sanità** (pipeline di pseudonimizzazione, air-gapped), **finanza**
(audit append-only, redazione campi all'ingestion, data center DORA), **PA** ("high-risk system
documentation pack", licenza perpetua procurement-friendly). Modello: licenza perpetua +
manutenzione. Casa madre: **I3K Technologies (i3k.eu)**.

**Perché è centrale per la call:** è **il nostro prodotto applicativo** e **usa già EuLLM**. La
narrazione: EULLM è il *livello modelli+inferenza sovrano*, RAG Enterprise è il *livello
applicativo sovrano*. Le schede di conformità del Hub sono l'input al pacchetto documentale di
RAG Enterprise. **Integrazione verticale = fossato competitivo.**

**Vantaggi di conformità (reali, difendibili):** self-hosted ⇒ I3K è **fornitore di software,
non responsabile del trattamento**; niente trasferimenti transatlantici ("no Schrems II
exposure" — vero *se* deployment realmente UE/air-gapped); erasure trattabile sul vector store;
audit "compatibile Art. 30 GDPR".

**Claim da rifinire (igiene sul nostro stesso prodotto — utile mostrare consapevolezza a Etel):**
- "GDPR ready" / "EU AI Act ready" sono **posizionamenti, non certificazioni** → dire "progettato
  per supportare la conformità".
- "No Schrems II exposure" è vero **solo** se i backup restano UE: attenzione al **backup rclone
  verso 70+ provider** — un bucket USA reintrodurrebbe silenziosamente un trasferimento. Aggiungere
  guardrail (target di backup solo UE).
- Il "high-risk documentation pack" va venduto come *"documentazione a supporto degli obblighi
  del deployer"*, **non** come autocertificazione: gli obblighi Art. 26 e la valutazione di
  conformità restano del **deployer**.
- Lista sicurezza attuale ("TLS + bcrypt + sessioni") **troppo sottile** per l'enterprise:
  aggiungere cifratura at-rest, key management, backup cifrati, vuln-management, postura pen-test
  (serve anche per Art. 32 GDPR e Art. 15 AI Act).
- "DORA-compliant data center" = proprietà del cliente/host, non del software (vedi §5.2).

---

## 7. Le schede di conformità — come organizzarle (il cuore della richiesta)

> **Precisazione fondamentale (come richiesto):** le schede si riferiscono ai **modelli
> verticalizzati** che produrremo (es. `eullm/legal-it-7b`), **non** ai modelli scaricabili oggi.
> "Scheda di conformità" è **un nostro termine di prodotto**: ufficialmente essa *impacchetta*
> gli artefatti che l'AI Act richiede, così che il **deployer** possa assolvere i **propri**
> obblighi. Non è un timbro di conformità del modello.

### 7.1 Cosa deve contenere (mappatura agli articoli)
La scheda è il contenitore di:
1. **Descrizione generale (Annex XI §1a):** compiti previsti e tipi di sistema in cui si integra;
   politica d'uso accettabile; data di rilascio e modalità di distribuzione; **architettura e n.
   di parametri**; modalità/formato input-output; **licenza**.
2. **Sviluppo (Annex XI §1b):** mezzi tecnici di integrazione; scelte di design e razionale;
   **dati di training** — tipo, provenienza, metodologie di curation/cleaning/filtering, n. di
   data point, misure anti-bias; **compute e tempo di training**; **consumo energetico** stimato.
3. **Riassunto pubblico dei contenuti di training (Art. 53(1)(d))** con il **template ufficiale
   AI Office** (24 lug 2025) — *obbligatorio anche open source*.
4. **Policy sul copyright / opt-out TDM (Art. 53(1)(c))** — *obbligatoria anche open source*.
5. **Note di trasparenza (Art. 50):** il modello genera contenuti sintetici → istruzioni al
   deployer su disclosure e marcatura.
6. **Delimitazione dell'uso previsto + trigger di alto rischio:** "destinato a…"; "**non**
   destinato a \[Annex III §…] senza valutazione di conformità del deployer"; lingue; limiti noti.
7. **Provenienza e integrità:** base model + licenza, catena di trasformazioni Forge
   (prune/distill/quant/LoRA), hash/fingerprint del GGUF, versione, data.
8. **(Opzionale, valore aggiunto) High-risk enablement pack** per i deployer in usi Annex III:
   pre-compila parti di Annex IV + Art. 26 (data governance, sorveglianza umana, logging,
   metriche di accuratezza, cybersecurity) — restando l'utilizzatore il soggetto responsabile.

### 7.2 Principi di design (come cambiare l'attuale scheda statica)
- **Per-modello, non statica:** ogni campo deriva dalla **provenienza reale** del modello,
  **generata da Forge** al termine della pipeline (oggi assente → da costruire).
- **Doppio formato:** JSON machine-readable (per Hub/RAG/audit) + Markdown human-readable (per
  legali e procurement).
- **Versionata, firmata, hash-ata:** ogni scheda legata al digest del GGUF; storicizzata.
- **Framing di responsabilità esplicito nel documento:** *"Queste informazioni servono ad
  abilitare la conformità del deployer; non costituiscono una valutazione di conformità né una
  garanzia. La classificazione di rischio dipende dall'uso."*
- **Niente asserzioni indifendibili:** eliminare `gdpr_compliant=true` secco, `personal_data="No
  personal data"` e `systemic_risk=false` hardcoded; sostituirli con campi *fattuali e
  qualificati* (es. `training_data_personal_data: "pseudonymised court rulings; see LIA ref";
  systemic_risk_flops_estimate: "<10^25 (not systemic)"`).

### 7.3 Struttura a due livelli
- **Livello A — "GPAI Model Compliance Card"** (sempre, per ogni modello verticalizzato):
  Annex XI §1 + training summary + copyright policy + trasparenza + uso previsto + provenienza.
- **Livello B — "High-Risk Deployment Enablement Pack"** (opzionale, per verticale sensibile):
  pre-compilazione Annex IV / Art. 26 a supporto del deployer.

*(Il template concreto — schema JSON + esempio compilato per `legal-it-7b` + checklist — è nel
file affiancato: `docs/compliance/compliance-card-template.md`.)*

---

## 8. Cosa facciamo NOI vs cosa fa l'UTILIZZATORE (ripartizione delle responsabilità)

Principio: **noi = provider di modello + abilitatore**; **utilizzatore = deployer** (ed
eventualmente provider del sistema ad alto rischio finale). Diamo gli strumenti; la conformità
del sistema nel contesto d'uso è dell'utilizzatore.

| Obbligo AI Act/GDPR | Chi risponde | Strumento che forniamo NOI |
|---|---|---|
| Riassunto pubblico dati di training (53(1)(d)) | **EULLM** (provider modello) | Generatore in Forge + pubblicazione su Hub |
| Policy copyright / opt-out TDM (53(1)(c)) | **EULLM** | Policy allegata alla scheda |
| Doc tecnica GPAI (Annex XI) | EULLM (ridotta se open source) | Scheda Livello A |
| Trasparenza Art. 50 (disclosure IA) | **Deployer** (chi espone il chatbot) | Feature disclosure nell'Engine + istruzioni in scheda |
| Marcatura contenuti sintetici (50(2)) | Provider del sistema generativo | (roadmap) watermark/provenance nell'Engine |
| Classificazione alto rischio (Art. 6) | **Deployer** (dipende dall'uso) | Scheda che delimita l'uso + trigger di rischio |
| Gestione rischio, data governance, Annex IV (Art. 9–11) | **Deployer/provider del sistema** | High-Risk Enablement Pack (Livello B) pre-compilato |
| Sorveglianza umana (Art. 14) | **Deployer** | Design human-in-the-loop + linee guida |
| Logging ≥6 mesi (Art. 12/26) | **Deployer** | Audit trail dell'Engine (da estendere: export/report) |
| DPIA (Art. 35 GDPR) | **Deployer/titolare** | Doc a supporto + audit compatibile Art. 30 |
| Base giuridica / Art. 9 / Art. 22 | **Deployer/titolare** | Anonimizzatore Forge + human-in-the-loop + guida basi giuridiche |
| Cancellazione dati (Art. 17) | **Deployer/titolare** | RAG (cancellabile) + minimizzazione PII nel training |
| Sicurezza (Art. 32 GDPR / Art. 15 AI Act) | Entrambi | On-prem + hardening + (roadmap) SBOM/CRA |

**Da dire a Etel:** questa tabella *è* il modello di business — ogni riga "utilizzatore" è un
punto in cui vendiamo lo strumento che gli fa spuntare la casella, senza ereditarne la
responsabilità.

---

## 9. Roadmap — da oggi al 2 dicembre 2027

**Ora → 2 ago 2026 (quick wins, execution visibile):**
1. **Disclosure Art. 50** nell'Engine (banner/header "sistema IA"). Piccolo, obbligatorio, subito.
2. **Correggere le affermazioni sovradimensionate** (README "every request/response";
   Hub "all data stays within EU borders") — allineare parole e codice.
3. **Generatore di scheda per-modello in Forge** (Livello A) + **de-hardcodare le schede del Hub**.
4. Per i primi modelli verticalizzati: **riassunto training (template) + policy copyright**.

**2026 → 2 dic 2027 (verso l'alto rischio):**
5. **High-Risk Enablement Pack** (Livello B) per legal/medical/finance.
6. **Audit trail 2.0:** export/report, provenienza (base model, adapter, fingerprint, versione),
   opzione di logging del contenuto *configurabile e on-prem* per chi ne ha bisogno (Art. 12/26).
7. **Marcatura contenuti sintetici (Art. 50(2))** — watermark/provenance (scadenza 2 dic 2026).
8. **Residenza UE reale** per la distribuzione dei modelli (mirror UE vs `huggingface.co`).
9. **CRA/PLD**: SBOM, secure-by-design, politica di aggiornamento/EOL per l'edizione commerciale;
   confine netto con la community edition non-commerciale.

**Milestone di posizionamento per il fondo:** ogni item sopra è un *proof point* di "compliance
by design" e un *pezzo di prodotto vendibile* — non solo costo di conformità.

---

## 10. Cheat-sheet per la call

**I tre messaggi:**
1. *Stack sovrano a due livelli* (EULLM + RAG Enterprise), **già funzionante** in inferenza,
   audit e anonimizzazione — con gap chiari che sono la nostra roadmap finanziata.
2. *Compliance-as-a-product*: vendiamo gli **strumenti** per mettersi in regola (schede, audit,
   on-prem), non ci assumiamo la responsabilità del deployer.
3. *Timing*: l'Omnibus sposta l'alto rischio a **dic 2027** → **~16 mesi** per arrivare con il
   prodotto pronto, mentre il mercato UE (PA/legale/sanità/finanza) *deve* comprare sovrano.

**Domande probabili di Etel e risposte pronte:**
- *"Ma l'AI Act non è già in vigore?"* → "Le pratiche vietate e gli obblighi GPAI sì; l'alto
  rischio è stato **rinviato al 2 dic 2027** dal Digital Omnibus di giugno. Il che è un
  vantaggio di go-to-market per noi."
- *"Vi assumete responsabilità legali sui modelli?"* → "No. Siamo provider di modelli aperti +
  abilitatori. La conformità è del sistema e del deployer; noi forniamo la documentazione e
  l'audit che glielo rendono possibile. Il README lo dichiara esplicitamente."
- *"Cosa avete già in produzione?"* → "Engine con audit on-by-default e zero telemetria,
  anonimizzatore PII in Forge, Hub con formato scheda e API. I modelli demo e le schede
  per-modello sono in sviluppo — è parte di ciò che finanziamo."
- *"Perché non basta un modello USA con un layer di compliance?"* → "Perché su cloud USA il Capo
  V GDPR e il rischio Schrems III non si tolgono con un layer; noi togliamo l'egress alla radice
  (on-prem) e forniamo la documentazione AI Act nativa."
- *"Qual è il rischio regolatorio per voi?"* → "Il test dell'1/3-compute (Forge potrebbe renderci
  provider GPAI del modello modificato): lo gestiamo documentando come se lo fossimo. E la linea
  commerciale/non-commerciale OSS per CRA/PLD, che gestiamo nei termini di licenza."

**L'ask:** \[da completare con la cifra/obiettivo del round] — capitale per: (a) completare i
primi 3 modelli verticalizzati con schede per-modello, (b) audit 2.0 + trasparenza Art. 50,
(c) go-to-market PA/regolati UE prima della scadenza dic 2027.

---

## 11. Avvertenze di accuratezza (per non farsi sorprendere)
- Il testo definitivo dell'Omnibus in Gazzetta è di pochissime settimane fa: le nuove date
  (2 dic 2027 / 2 ago 2028) convergono in tutte le analisi, ma "la GU è l'autorità ultima".
- Le **Linee guida GPAI**, il **Code of Practice** e il **template training-summary** sono
  *soft-law* della Commissione, non il Regolamento.
- Il **test dell'1/3-compute** è **indicativo**, non una soglia netta.
- La **Data Omnibus lato GDPR** è **solo proposta**: non costruirci compliance.

---

## 12. Fonti principali
**UE ufficiali:** Consiglio, comunicato 29 giu 2026 (via libera all'Omnibus);
Parlamento UE — Legislative Train "Digital Omnibus on AI"; Commissione — digital-strategy
(Code of GPAI, template training); AI Act Service Desk (timeline, Art. 111); testo primario
EUR-Lex 2024/1689; testo articoli/allegati: artificialintelligenceact.eu.
**Omnibus (analisi legali):** Gibson Dunn; White & Case; Freshfields (#34); Jones Walker;
DLA Piper.
**GPAI / open source / modifica:** Hugging Face (OS & GPAI); DLA Piper (GPAI Guidelines);
artificialintelligenceact.eu (Modifying AI).
**GDPR / EDPB:** EDPB Opinion 28/2024 (PDF); EDPB "AI Privacy Risks & Mitigations — LLMs"
(10 apr 2025); CNIL (feb/giu/lug 2025); IAPP; Norton Rose; sentenza CGUE SCHUFA C-634/21.
**Altre norme:** PLD Dir. 2024/2853 (EUR-Lex; Gibson Dunn; Reed Smith); CRA Reg. 2024/2847
(EC; OpenSSF; BCLP); DORA Reg. 2022/2554; NIS2 Dir. 2022/2555 (tracker ECSO); Data Act Reg.
2023/2854; DGA Reg. 2022/868; ritiro AI Liability Directive (Bird & Bird; IAPP; EAPIL).
**DPF / trasferimenti:** DLA Piper; activeMind (DPF & US Supreme Court).
**rag-enterprise.com:** sito ufficiale (home/features/enterprise); i3k.eu.

*(URL completi nei materiali di ricerca allegati alla preparazione; disponibili su richiesta.)*
