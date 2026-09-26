# Microsoft Store submission: EuLLM

The text for each Partner Center field of the EuLLM submission, ready to
paste. Keep it in step with the product: when a release changes what EuLLM
does, update it here first, then in Partner Center.

## Pricing and availability

- Markets: all
- Audience: public; discoverable in the Store
- Base price: **Free**

## Properties

- Category: **Developer tools** (subcategory: none)
- Privacy policy URL: `https://github.com/eullm/eullm/blob/main/docs/privacy.md` (EuLLM's own policy, versioned with the code; the Store requires one for every Win32 app)
- Website: `https://eullm.eu`
- Support contact: `info@i3k.eu`
- System requirements (recommended): 16 GB RAM; for `eullm-cuda`, an NVIDIA
  RTX 3000, 4000 or 5000 GPU with driver 580 or newer

## Age ratings

Answer the questionnaire as an app, not a game. Where it asks about
content users can generate or access: EuLLM runs AI language models that
produce text the user asks for, and it has no content filter of its own;
with the `--web` option it fetches web pages the user names. Answer those
questions yes, and no to user-to-user interaction, location sharing and
purchases, which it does not have.

## Packages

Upload `eullm-windows-x64-store.msix` from the GitHub release
(`https://github.com/eullm/eullm/releases`). Each new release: a new
submission with that release's `.msix`.

## Store listing: English (en-us)

### Description

EuLLM runs large language models on your own computer. Download an open model in GGUF format, from Hugging Face or from the EuLLM catalog, and chat with it, or serve it to your own applications, without sending a single word to the cloud.

EuLLM is a single program you use from the terminal. It speaks the same API as Ollama and OpenAI on http://localhost:11434, so the tools and libraries you already use work unchanged, and it opens a built-in chat in your browser at http://localhost:11435.

Private by design: no telemetry, no account, nothing leaves your machine except the model downloads you ask for. Every request can be written to a local audit log, which helps organisations document their AI use under the EU AI Act.

Two commands are installed:
• eullm runs on the CPU, on any 64-bit Windows PC.
• eullm-cuda uses an NVIDIA GeForce RTX 3000, 4000 or 5000 graphics card (driver 580 or newer) and is much faster.

Getting started, in Windows Terminal or PowerShell:
eullm run hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M

EuLLM is open source (AGPL-3.0) and developed in Italy by I3K Technologies. Source code, documentation and the Linux and macOS versions: https://github.com/eullm/eullm

### Short description

Run large language models locally on Windows: private, offline, Ollama- and OpenAI-compatible.

### Features

- Runs open GGUF models (Qwen, Mistral, Gemma, DeepSeek, Phi and more) on your own PC
- Download any GGUF model straight from Hugging Face with one command
- Ollama- and OpenAI-compatible API on localhost:11434
- Built-in chat in your browser at localhost:11435
- NVIDIA GPU acceleration with eullm-cuda (RTX 3000/4000/5000)
- No telemetry, no account: your prompts never leave your machine
- Local audit log for EU AI Act record-keeping
- Continuous batching: serves many requests in parallel
- Open source, AGPL-3.0

### Search terms (7 max)

local AI · offline LLM · Ollama · GGUF · private AI · LLM server · AI chat

### Notes for certification (Submission options)

EuLLM is a console application: after installing, open Windows Terminal or PowerShell and run `eullm --version`. To try it end to end, run `eullm run hf.co/Qwen/Qwen3-0.6B-GGUF:Q8_0`, which downloads a 640 MB open model (Apache 2.0) from Hugging Face and starts it: the chat opens in the browser at http://localhost:11435 and the HTTP API answers on http://localhost:11434 (add `--cli` to chat in the terminal instead). Running `eullm` with no arguments opens an interactive model picker. `eullm-cuda` is the same program built for NVIDIA RTX 3000/4000/5000 GPUs with driver 580 or newer; on a machine without such a GPU please test with `eullm`.

### Restricted capability justification (runFullTrust)

EuLLM is a Win32 console application (a local LLM inference engine and HTTP server) packaged as MSIX. runFullTrust is required to run its classic desktop executables, which are invoked from the terminal through the execution aliases `eullm` and `eullm-cuda`.

## Store listing: Italian (it-it)

### Description

EuLLM esegue modelli linguistici di grandi dimensioni sul tuo computer. Scarichi un modello aperto in formato GGUF, da Hugging Face o dal catalogo EuLLM, e ci chatti o lo metti a disposizione delle tue applicazioni, senza mandare una sola parola al cloud.

EuLLM è un unico programma che si usa dal terminale. Parla la stessa API di Ollama e di OpenAI su http://localhost:11434, quindi gli strumenti e le librerie che già usi funzionano senza modifiche, e apre una chat integrata nel browser su http://localhost:11435.

Privato per costruzione: niente telemetria, niente account, nulla lascia il tuo computer tranne i download dei modelli che chiedi tu. Ogni richiesta può essere registrata in un log di audit locale, utile alle organizzazioni per documentare l'uso dell'IA secondo l'AI Act europeo.

Vengono installati due comandi:
• eullm funziona sulla CPU, su qualsiasi PC Windows a 64 bit.
• eullm-cuda usa una scheda video NVIDIA GeForce RTX 3000, 4000 o 5000 (driver 580 o successivo) ed è molto più veloce.

Per iniziare, in Terminale Windows o PowerShell:
eullm run hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M

EuLLM è open source (AGPL-3.0) ed è sviluppato in Italia da I3K Technologies. Codice sorgente, documentazione e versioni per Linux e macOS: https://github.com/eullm/eullm

### Short description

Modelli linguistici in locale su Windows: privati, offline, compatibili con Ollama e OpenAI.

### Features

- Esegue modelli GGUF aperti (Qwen, Mistral, Gemma, DeepSeek, Phi e altri) sul tuo PC
- Scarica qualsiasi modello GGUF da Hugging Face con un solo comando
- API compatibile con Ollama e OpenAI su localhost:11434
- Chat integrata nel browser su localhost:11435
- Accelerazione su GPU NVIDIA con eullm-cuda (RTX 3000/4000/5000)
- Niente telemetria, niente account: i tuoi prompt non lasciano il tuo computer
- Log di audit locale per la documentazione richiesta dall'AI Act
- Continuous batching: serve molte richieste in parallelo
- Open source, AGPL-3.0

### Search terms (7 max)

IA locale · LLM offline · Ollama · GGUF · IA privata · server LLM · chat IA

## Screenshots

At least one, PNG, 1366×768 or larger, taken on Windows. Suggested:

1. Windows Terminal running `eullm run hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M`
   mid-answer.
2. The browser chat at `http://localhost:11435/` with a conversation.
3. The interactive model picker (`eullm` with no arguments).
4. An existing Ollama or OpenAI client pointed at `localhost:11434`.
