# EuLLM Engine — Privacy Policy / Informativa sulla privacy

*Last updated / Ultimo aggiornamento: 26 September 2026*

[English](#english) · [Italiano](#italiano)

---

## English

This policy covers the EuLLM Engine software (the `eullm` program, including
the `eullm` and `eullm-cuda` commands installed from the Microsoft Store and
the downloads published on GitHub). It is published by I3K Technologies
S.R.L., Via Enrico Cosenz 54, 20158 Milano, Italy — `info@i3k.eu`.

### In short

EuLLM runs on your computer. **I3K Technologies does not receive, collect or
store any of your data through EuLLM**: there is no telemetry, no analytics,
no crash reporting and no account. Your prompts and the model's answers are
processed on your machine and are not sent to us or to anyone else.

### What stays on your computer

- **Models** you download, in `%USERPROFILE%\.eullm\models` on Windows
  (`~/.eullm/models` on Linux and macOS), or in the folder set by
  `EULLM_MODELS_DIR`.
- **Audit log**, in `%USERPROFILE%\.eullm\audit\audit.jsonl` (or the folder
  set by `EULLM_AUDIT_DIR`). It records metadata about each request — time,
  model, request type, token counts, duration and, if you configured API
  keys, which key was used. It does **not** contain the text of prompts or
  answers. It exists so that organisations can document their use of AI
  (EU AI Act) and it never leaves your computer.
- **Prompts and answers** are held in memory while EuLLM runs. EuLLM does
  not write them to disk. The browser chat keeps an API key, if you use one,
  only for the open tab (`sessionStorage`).

You control all of this: delete the `.eullm` folder to remove models and the
audit log. Uninstalling EuLLM does not delete that folder.

### When EuLLM connects to the internet

Only when you ask it to, and only to the services involved:

| When | Service | What they receive |
|---|---|---|
| You download a model from Hugging Face (`eullm pull` / `eullm run hf.co/…`) | Hugging Face (`huggingface.co` and its download servers) | Your IP address, the model requested, and the program version (`eullm/<version>`). If you set `HF_TOKEN`, the token is sent to `huggingface.co` only. |
| You open the model catalog (picker or browser chat) | GitHub (`raw.githubusercontent.com`) | Your IP address and the program version. |
| You install or update with `install.ps1` / `install.sh`, or download a release | GitHub | Your IP address. |
| You run with `--web` and a message contains a web address | The website you named | Your IP address and a standard browser-style request. |
| You install or update through the Microsoft Store | Microsoft | As described in Microsoft's own privacy statement. |

Each of these services processes that information under its own privacy
policy; I3K Technologies does not receive it. EuLLM sends nothing when you
use a model that is already on your computer, and it can run completely
offline.

### Children

EuLLM is a developer tool and is not directed at children.

### Contact and your rights

Because EuLLM sends no personal data to I3K Technologies, we hold none about
you from the software. If you write to us (`info@i3k.eu`), your message is
handled under the I3K Technologies privacy policy at
`https://www.i3k.eu/privacy`, which also explains your rights under the
GDPR, including the right to complain to the Italian supervisory authority
(Garante per la protezione dei dati personali).

### Changes

We update this policy when a new EuLLM feature changes what it connects to
or stores. The date at the top says when it last changed; every version is
kept in the history of this file on GitHub.

---

## Italiano

Questa informativa riguarda il software EuLLM Engine (il programma `eullm`,
compresi i comandi `eullm` ed `eullm-cuda` installati dal Microsoft Store e
i download pubblicati su GitHub). È pubblicata da I3K Technologies S.R.L.,
Via Enrico Cosenz 54, 20158 Milano — `info@i3k.eu`.

### In breve

EuLLM funziona sul tuo computer. **I3K Technologies non riceve, non raccoglie
e non conserva alcun tuo dato attraverso EuLLM**: niente telemetria, niente
statistiche d'uso, niente segnalazioni di crash, nessun account. I tuoi
prompt e le risposte del modello sono elaborati sul tuo computer e non
vengono inviati né a noi né ad altri.

### Cosa resta sul tuo computer

- **I modelli** che scarichi, in `%USERPROFILE%\.eullm\models` su Windows
  (`~/.eullm/models` su Linux e macOS), oppure nella cartella indicata da
  `EULLM_MODELS_DIR`.
- **Il log di audit**, in `%USERPROFILE%\.eullm\audit\audit.jsonl` (oppure
  nella cartella indicata da `EULLM_AUDIT_DIR`). Registra i metadati di ogni
  richiesta: ora, modello, tipo di richiesta, numero di token, durata e, se
  hai configurato chiavi API, quale chiave è stata usata. **Non** contiene il
  testo dei prompt né delle risposte. Serve alle organizzazioni per
  documentare l'uso dell'IA (AI Act) e non lascia mai il tuo computer.
- **Prompt e risposte** restano in memoria mentre EuLLM è in esecuzione.
  EuLLM non li scrive su disco. La chat nel browser conserva l'eventuale
  chiave API solo per la scheda aperta (`sessionStorage`).

Hai il pieno controllo: cancellando la cartella `.eullm` rimuovi modelli e
log di audit. La disinstallazione di EuLLM non cancella quella cartella.

### Quando EuLLM si collega a internet

Solo quando lo chiedi tu, e solo ai servizi coinvolti:

| Quando | Servizio | Cosa riceve |
|---|---|---|
| Scarichi un modello da Hugging Face (`eullm pull` / `eullm run hf.co/…`) | Hugging Face (`huggingface.co` e i suoi server di download) | Il tuo indirizzo IP, il modello richiesto e la versione del programma (`eullm/<versione>`). Se imposti `HF_TOKEN`, il token viene inviato solo a `huggingface.co`. |
| Apri il catalogo dei modelli (menu di scelta o chat nel browser) | GitHub (`raw.githubusercontent.com`) | Il tuo indirizzo IP e la versione del programma. |
| Installi o aggiorni con `install.ps1` / `install.sh`, o scarichi una release | GitHub | Il tuo indirizzo IP. |
| Usi `--web` e un messaggio contiene un indirizzo web | Il sito indicato | Il tuo indirizzo IP e una normale richiesta come quella di un browser. |
| Installi o aggiorni dal Microsoft Store | Microsoft | Quanto descritto nell'informativa di Microsoft. |

Ciascuno di questi servizi tratta quelle informazioni secondo la propria
informativa; I3K Technologies non le riceve. Se usi un modello già presente
sul tuo computer EuLLM non invia nulla, e può funzionare completamente
offline.

### Minori

EuLLM è uno strumento per sviluppatori e non è rivolto ai minori.

### Contatti e diritti

Poiché EuLLM non invia dati personali a I3K Technologies, non ne
conserviamo alcuno che ti riguardi tramite il software. Se ci scrivi
(`info@i3k.eu`), il messaggio è trattato secondo l'informativa privacy di
I3K Technologies, `https://www.i3k.eu/privacy`, che descrive anche i tuoi
diritti secondo il GDPR, compreso il diritto di reclamo al Garante per la
protezione dei dati personali.

### Modifiche

Aggiorniamo questa informativa quando una nuova funzione di EuLLM cambia ciò
a cui si collega o ciò che conserva. La data in alto indica l'ultima
modifica; ogni versione resta nella cronologia di questo file su GitHub.
