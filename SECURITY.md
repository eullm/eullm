# Security Policy

## Reporting a vulnerability

Please report security issues privately, **not** as a public issue:

- Use GitHub's [private vulnerability reporting](https://github.com/eullm/eullm/security/advisories/new), or
- email **security@eullm.eu**

Useful things to include: the version (`eullm -V`), the platform and build
variant (CPU / CUDA / Metal / ROCm / Vulkan), how the engine was started
(`run` / `serve`, flags), and the smallest reproduction you have.

You can expect an acknowledgement within **5 working days** and an assessment
within **15**. If we accept the report we will agree a disclosure date with you
and credit you in the release notes unless you prefer otherwise. If we decline
it we will explain why, so you can push back if you think we are wrong.

## Supported versions

The engine is pre-1.0 and moves fast. Security fixes go into the **latest
release only** — there are no maintained release branches. If you are running an
older version, upgrading is the fix.

| Version        | Supported          |
| -------------- | ------------------ |
| latest `0.6.x` | :white_check_mark: |
| anything older | :x: — upgrade      |

## Threat model — please read before reporting

The engine's default posture is **a single trusted machine**. Knowing what is
and isn't a boundary saves everyone time:

- **There is no authentication.** Any client allowed to reach the port can use
  the API, swap the loaded model, or unload it. This is deliberate for the local
  single-user case, and is why the listener is restricted by source IP instead.
- **The source-IP allowlist is not access control on an untrusted network.** It
  defaults to loopback only and is widened with `EULLM_ALLOWED_IPS`. It cannot
  express two cases: behind Docker's published ports every external client is
  NAT-ed to the bridge gateway address, and a request from the user's own
  browser genuinely originates from loopback. Treat it as a convenience
  boundary for a trusted LAN.
- **`--web` fetches URLs found in prompts.** With it enabled, whoever can send a
  prompt can make the engine issue outbound GET requests. Don't enable it on a
  host with reachable internal services you wouldn't want fetched.
- **A GGUF file is executable content in practice.** It is memory-mapped into a
  process running llama.cpp. Only load models you would trust as a binary.
  Catalog pulls are verified against a recorded SHA-256; off-catalog pulls
  (`hf.co/...`, direct URLs) are not, because there is no digest to check
  against.
- **The audit trail is a local record, not a tamper-proof one.** It is an
  append-only JSONL file with no hash chain and no signature, so anyone with
  write access to it can alter it undetectably.

Reports that one of the documented behaviours above exists are welcome as
**hardening suggestions** — several are tracked in
[`docs/backlog-fix-e-hardening.md`](docs/backlog-fix-e-hardening.md) — but they
aren't treated as vulnerabilities, because nothing here claims otherwise.
Anything that breaks a boundary we *do* claim is a vulnerability, and we want to
hear about it.
