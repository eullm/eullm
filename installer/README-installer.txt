EuLLM Engine — sovereign LLM runtime for Europe
=================================================

Thanks for installing EuLLM!

WHAT YOU GOT
------------
- eullm.exe           — the engine binary (Ollama + OpenAI compatible)
- Embedded chat UI    — opens at http://localhost:11435/ when the engine runs
- Start Menu shortcut "EuLLM Chat" — launches everything for you

QUICK START
-----------
1. Get a GGUF model. Recommended (Apache 2.0, multilingual):
     https://huggingface.co/Qwen/Qwen3-8B-GGUF
   Download Qwen3-8B-Q4_K_M.gguf (~4.7 GB) and drop it into:
     %LOCALAPPDATA%\EuLLM\models\

2. Click "EuLLM Chat" in the Start Menu. The engine starts, your browser
   opens, and you can chat — fully local, fully sovereign.

   Alternative: open a terminal and run:
     eullm run path\to\your-model.gguf
   then visit http://localhost:11435/ in any browser.

OLLAMA / OPENAI CLIENTS
-----------------------
EuLLM speaks both APIs on the same port:
  http://localhost:11434/api    (Ollama-compatible)
  http://localhost:11434/v1     (OpenAI-compatible)

Any existing client (Open WebUI, LangChain, n8n, OpenAI SDK, ...) works
without code changes.

AUDIT TRAIL (AI Act)
--------------------
Every request and response is logged locally to:
  %USERPROFILE%\.eullm\audit\audit.jsonl

Nothing is ever sent to non-EU servers. No telemetry, no analytics,
no crash reports.

WINDOWS SMARTSCREEN
-------------------
On first launch, Windows may show "Windows protected your PC" because
the binary isn't yet code-signed. Click "More info" -> "Run anyway".
Code signing is on the roadmap.

UNINSTALL
---------
Settings -> Apps -> "EuLLM Engine" -> Uninstall
(Or: Start Menu -> EuLLM -> Uninstall EuLLM)

LINKS
-----
Website:  https://eullm.eu
Source:   https://github.com/eullm/eullm
License:  Apache 2.0 (see LICENSE next to this file)
DOI:      10.5281/zenodo.20412979
