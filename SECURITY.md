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

- **Authentication is off unless you configure it.** With no `EULLM_API_KEYS`
  set, any client allowed to reach the port can use the API, swap the loaded
  model, or unload it. That is deliberate for the local single-user case, and is
  why the listener is restricted by source IP instead. Set
  `EULLM_API_KEYS=id:secret[:rpm=N]` (or point `EULLM_API_KEYS_FILE` at a file
  with one entry per line) to require a bearer token — **necessary for any
  deployment where the caller is not the operator.** Note the resulting posture:
  a valid key admits a request from any source address, because behind Docker's
  address translation an address-based check cannot discriminate between callers
  anyway. The key id is recorded in every audit entry.
- **The source-IP allowlist is not access control on an untrusted network.** It
  defaults to loopback only and is widened with `EULLM_ALLOWED_IPS`. It cannot
  express two cases: behind Docker's published ports every external client is
  NAT-ed to the bridge gateway address, and a request from the user's own
  browser genuinely originates from loopback. Treat it as a convenience boundary
  for a trusted LAN; use API keys for anything more.
- **Browser origins are restricted, but only browsers are.** By default any
  loopback origin is allowed (so the bundled chat UI and a local frontend work),
  a cross-origin request with side effects is refused with 403, and
  `EULLM_ALLOWED_ORIGINS` configures the rest. This constrains pages running in
  the user's browser. It is no constraint at all on a program, which can send
  any headers it likes — that is what API keys are for.
- **`--web` fetches URLs found in prompts.** With it enabled, whoever can send a
  prompt can make the engine issue outbound GET requests. The fetcher refuses
  non-`https` schemes, refuses hosts that resolve to loopback, private,
  link-local or cloud-metadata addresses, re-validates every redirect hop, caps
  the body, and accepts only textual content types;
  `EULLM_WEB_ALLOWED_DOMAINS` narrows it to an allowlist of sources.
  `EULLM_WEB_ALLOW_PRIVATE_HOSTS=1` turns the address check off for intranet
  use — with it set, treat `--web` as equivalent to handing prompt authors a
  GET primitive on your internal network.
- **A GGUF file is executable content in practice.** It is memory-mapped into a
  process running llama.cpp. Only load models you would trust as a binary.
  Catalog pulls are verified against a recorded SHA-256; off-catalog pulls
  (`hf.co/...`, direct URLs) are not, because there is no digest to check
  against.
- **The audit trail is a local record, not a tamper-proof one.** It is an
  append-only JSONL file with no hash chain and no signature, so anyone with
  write access to it can alter it undetectably.
- **A request's `model` field cannot name an arbitrary path** unless
  `EULLM_ALLOW_MODEL_PATHS=1` is set. Without it, names resolve inside the model
  store or a deliberate container mount point only, and an existing file is
  indistinguishable from a missing one in the error returned. With it set, the
  field becomes a way for any caller to point the loader at any file this process
  can read.

Reports that one of the documented behaviours above exists are welcome as
**hardening suggestions** — several are tracked in
[`docs/backlog-fix-e-hardening.md`](docs/backlog-fix-e-hardening.md) — but they
aren't treated as vulnerabilities, because nothing here claims otherwise.
Anything that breaks a boundary we *do* claim is a vulnerability, and we want to
hear about it.
