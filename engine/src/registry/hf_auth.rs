//! `HF_TOKEN`: authenticated requests to Hugging Face, and to nothing else.
//!
//! Gated and private repositories answer an anonymous download with 401.
//! When `HF_TOKEN` is set, every request the registry makes to Hugging Face
//! carries it as a bearer token: the API (model info, file tree, search) and
//! every download — the range probe, each parallel range, the single-stream
//! fallback — for weights, shards and projectors alike, since they all go
//! through a [`RegistryClient`]. Unset, nothing changes: the same requests
//! go out as before, anonymous.
//!
//! Three rules keep the token where it belongs, each in one place:
//!
//! * **Per request, never a client default.** A default header would ride on
//!   every URL the client is handed, and `download_file` takes arbitrary
//!   ones. [`RegistryClient::get`] attaches the token only when
//!   [`is_hf_url`] says the address is Hugging Face's: `https`, the host
//!   exactly `huggingface.co`, the default port. Not a subdomain, not a
//!   look-alike, not plain `http`, and not `hf.co` either: it only redirects
//!   to `huggingface.co`, and a token sent to it would be dropped at that
//!   redirect, by the next rule. The client inside is private to this
//!   module, so nothing that holds a `RegistryClient` can make a request
//!   that skips the check.
//! * **Never across a redirect to another host.** A download is redirected
//!   from `huggingface.co` to a CDN (`us.aws.cdn.hf.co` and the like) with a
//!   pre-signed link, which needs no token. reqwest removes `Authorization`
//!   from a redirected request whenever the host or the port changes
//!   (`remove_sensitive_headers` in its `redirect.rs`); the promise above
//!   rests on that, so a test below holds reqwest to it.
//! * **Never printed or stored.** [`HfToken`] has no `Display`, its `Debug`
//!   is redacted, and reqwest marks the header sensitive. The token is never
//!   part of a URL, so the errors and log lines that name one cannot carry
//!   it, and it never reaches a manifest or the audit trail.
//!
//! It is read from the process environment, else from the `.env` file in the
//! folder eullm starts from (the rule of every setting, [`crate::settings`]),
//! under the name the Hugging Face tools and llama.cpp's `-hf` read. There is
//! no flag for it on purpose: a secret on a command line is in `ps` for every
//! local user.

use std::env::VarError;
use std::sync::OnceLock;

use reqwest::header::HeaderMap;
use reqwest::{RequestBuilder, StatusCode};

/// The variable the token is read from.
pub const HF_TOKEN_VAR: &str = "HF_TOKEN";

/// The one host the token is sent to.
const HF_HOST: &str = "huggingface.co";

/// Where a Hugging Face account makes its access tokens.
const TOKENS_PAGE: &str = "https://huggingface.co/settings/tokens";

/// A Hugging Face access token.
///
/// There is no `Display` and no accessor: the only thing ever done with it is
/// [`RegistryClient::get`] putting it in a request header.
#[derive(Clone, PartialEq, Eq)]
pub struct HfToken(String);

impl std::fmt::Debug for HfToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HfToken(<redacted>)")
    }
}

impl HfToken {
    /// The token, from what reading `HF_TOKEN` returned.
    ///
    /// Unset or blank is no token: anonymous, exactly as before this existed.
    /// Whitespace around it is trimmed, because `HF_TOKEN=$(cat file)` keeps
    /// the file's final newline. A value no HTTP header can carry — a space
    /// or a line break inside it, a non-ASCII character — is an error rather
    /// than no token, and it fails the requests the token was meant for:
    /// someone who set a token and silently got anonymous access is left
    /// reading a 401 about a token they know they set. The error never
    /// repeats the value.
    ///
    /// Pure, so it is tested without touching the process environment.
    pub fn resolve(raw: Result<String, VarError>) -> Result<Option<Self>, String> {
        let raw = match raw {
            Ok(raw) => raw,
            Err(VarError::NotPresent) => return Ok(None),
            Err(VarError::NotUnicode(_)) => return Err(unusable_token()),
        };
        let token = raw.trim();
        if token.is_empty() {
            return Ok(None);
        }
        if !token.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(unusable_token());
        }
        Ok(Some(Self(token.to_string())))
    }
}

fn unusable_token() -> String {
    format!(
        "{HF_TOKEN_VAR} is set, but not to anything an HTTP header can carry: it holds a \
         space, a line break or a non-ASCII character. Set it to the token alone."
    )
}

/// The process's token, read once.
///
/// Its presence and where it was set — never its value — are logged the
/// first time, so whether a download went out authenticated can be read
/// from the log.
pub fn hf_token() -> Result<Option<&'static HfToken>, String> {
    static TOKEN: OnceLock<Result<Option<HfToken>, String>> = OnceLock::new();
    TOKEN
        .get_or_init(|| {
            let (raw, source) = read_token();
            let token = HfToken::resolve(raw);
            if let Ok(Some(_)) = token {
                tracing::info!("{HF_TOKEN_VAR} is set ({source}): requests to {HF_HOST} carry it");
            }
            token
        })
        .as_ref()
        .map(Option::as_ref)
        .map_err(Clone::clone)
}

/// What reading `HF_TOKEN` returned, and where: the environment's value, else
/// the `.env` file's. A value in the environment that is not Unicode stays
/// the error it is rather than letting the file's take its place unseen.
fn read_token() -> (Result<String, VarError>, &'static str) {
    if let Err(VarError::NotUnicode(raw)) = std::env::var(HF_TOKEN_VAR) {
        return (Err(VarError::NotUnicode(raw)), "environment");
    }
    match crate::settings::var_with_source(HF_TOKEN_VAR) {
        Some((token, source)) => (Ok(token), source.label()),
        None => (Err(VarError::NotPresent), "environment"),
    }
}

/// Whether the token may be sent to `url`: `https`, the host exactly
/// [`HF_HOST`], the default port. The `url` crate lowercases the host and
/// drops an explicit `:443`, so neither changes the answer; a trailing dot,
/// a userinfo trick or a subdomain does, and gets no token.
pub fn is_hf_url(url: &str) -> bool {
    reqwest::Url::parse(url).is_ok_and(|url| {
        url.scheme() == "https" && url.host_str() == Some(HF_HOST) && url.port().is_none()
    })
}

/// How a request stood with the token, for explaining a refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenUse {
    /// `HF_TOKEN` is not set.
    Absent,
    /// It is set, and the request was addressed to Hugging Face: it carried it.
    Sent,
    /// It is set, but the request was addressed elsewhere and went without
    /// it — an `hf.co` or plain `http` address that redirects to Hugging Face.
    Withheld,
}

/// A `reqwest::Client` that adds the token to requests for Hugging Face, and
/// to nothing else. Every request the registry makes goes through one.
#[derive(Clone)]
pub struct RegistryClient {
    http: reqwest::Client,
    /// The token, or why `HF_TOKEN` could not be used. The error fails only
    /// the requests the token would have gone on: a download from any other
    /// address never needed it.
    token: Result<Option<HfToken>, String>,
}

impl RegistryClient {
    /// Wrap `http`, with the process's token when `HF_TOKEN` is set.
    pub fn new(http: reqwest::Client) -> Self {
        Self {
            http,
            token: hf_token().map(|token| token.cloned()),
        }
    }

    /// Wrap `http` with the given token, or none: for tests, which must not
    /// depend on whatever `HF_TOKEN` the machine running them has.
    #[cfg(test)]
    pub fn with_token(http: reqwest::Client, token: Option<HfToken>) -> Self {
        Self {
            http,
            token: Ok(token),
        }
    }

    /// A GET for `url`, carrying the token only when `url` is Hugging Face's.
    ///
    /// `Err` for a Hugging Face URL when `HF_TOKEN` is set to something
    /// unusable, rather than a request that quietly goes out anonymous.
    pub fn get(&self, url: &str) -> Result<RequestBuilder, String> {
        let request = self.http.get(url);
        if !is_hf_url(url) {
            return Ok(request);
        }
        match &self.token {
            Ok(Some(token)) => Ok(request.bearer_auth(&token.0)),
            Ok(None) => Ok(request),
            Err(unusable) => Err(unusable.clone()),
        }
    }

    /// How a request for `url` stands with the token.
    pub fn token_use(&self, url: &str) -> TokenUse {
        match (&self.token, is_hf_url(url)) {
            (Ok(None), _) => TokenUse::Absent,
            (Ok(Some(_)), true) => TokenUse::Sent,
            (Ok(Some(_)) | Err(_), false) => TokenUse::Withheld,
            // `get` refuses to build this request, so no answer to it is
            // ever explained; anonymous is what it would have been.
            (Err(_), true) => TokenUse::Absent,
        }
    }

    /// When `response` is Hugging Face refusing the request made for `url`,
    /// why, and what to do about it. See [`refusal_message`].
    pub fn refusal(&self, url: &str, response: &reqwest::Response) -> Option<String> {
        refusal_message(
            response.url().as_str(),
            response.status(),
            response.headers(),
            self.token_use(url),
        )
    }
}

/// Why Hugging Face turned a request away, and what to do about it.
///
/// `None` unless `status` is 401 or 403 *and* the answer came from Hugging
/// Face itself — `answered_by` is the response's final URL, after redirects:
/// a CDN refusing an expired pre-signed link says nothing about the token.
///
/// Hugging Face's own explanation, from its `x-error-message` header, is
/// quoted, because it is the one place that says "you are not in the
/// authorized list" or names a permission a fine-grained token lacks. Its
/// `x-error-code` tells a gated repository (`GatedRepo`) apart from one that
/// is private or does not exist, which it deliberately answers the same way
/// to anyone who cannot see it.
pub fn refusal_message(
    answered_by: &str,
    status: StatusCode,
    headers: &HeaderMap,
    token: TokenUse,
) -> Option<String> {
    if !matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
        || !is_hf_url(answered_by)
    {
        return None;
    }
    let header = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|v| !v.is_empty())
    };
    let gated = header("x-error-code") == Some("GatedRepo");
    let quoted = header("x-error-message")
        .map(|m| format!(": \"{}\"", clip(&crate::audit::sanitize_for_log(m), 300)))
        .unwrap_or_default();
    let repo = repo_of(answered_by);
    let page = repo.as_deref().map_or_else(
        || format!("its page on {HF_HOST}"),
        |repo| format!("https://{HF_HOST}/{repo}"),
    );
    let advice = match (token, status, gated) {
        (TokenUse::Withheld, ..) => format!(
            "{HF_TOKEN_VAR} is set, but it is only sent to https://{HF_HOST} addresses and this \
             request was addressed elsewhere. Use the https://{HF_HOST} address, or \
             hf.co/<owner>/<repo>[:<quant>]."
        ),
        (TokenUse::Absent, _, true) => format!(
            "The repository is gated. Accept its terms on {page} with your Hugging Face \
             account, then set {HF_TOKEN_VAR} to an access token of that account ({TOKENS_PAGE})."
        ),
        (TokenUse::Absent, ..) => format!(
            "The repository is private, or does not exist. If it is private, set \
             {HF_TOKEN_VAR} to an access token of an account that can read it ({TOKENS_PAGE})."
        ),
        (TokenUse::Sent, StatusCode::UNAUTHORIZED, _) => format!(
            "The token in {HF_TOKEN_VAR} was not accepted: it is invalid, expired or revoked, \
             or the repository does not exist. A new token can be made at {TOKENS_PAGE}."
        ),
        (TokenUse::Sent, _, true) => format!(
            "The repository is gated, and the account behind {HF_TOKEN_VAR} has no access to \
             it yet. Accept its terms, or request access, on {page} with that account; some \
             authors grant access by hand."
        ),
        (TokenUse::Sent, ..) => format!(
            "The token in {HF_TOKEN_VAR} is not allowed to read it: a fine-grained token needs \
             read access to this repository, or to public gated repositories, enabled in its \
             settings ({TOKENS_PAGE})."
        ),
    };
    let subject = repo.unwrap_or_else(|| answered_by.to_string());
    Some(format!(
        "Hugging Face refused {subject} (HTTP {}{quoted}). {advice}",
        status.as_u16()
    ))
}

/// `owner/repo` out of a Hugging Face URL: a download
/// (`/{owner}/{repo}/resolve/…`), the API (`/api/models/{owner}/{repo}…`), or
/// the cache a download is redirected through
/// (`/api/resolve-cache/models/{owner}/{repo}/…`).
fn repo_of(url: &str) -> Option<String> {
    let url = reqwest::Url::parse(url).ok()?;
    let segments: Vec<&str> = url.path_segments()?.collect();
    let (owner, repo) = match segments.as_slice() {
        ["api", "models", owner, repo, ..]
        | ["api", "resolve-cache", "models", owner, repo, ..]
        | [owner, repo, "resolve", ..] => (*owner, *repo),
        _ => return None,
    };
    let id = format!("{owner}/{repo}");
    super::is_plausible_repo_id(&id).then_some(id)
}

/// `text`, cut to at most `max` characters.
fn clip(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{AUTHORIZATION, HeaderValue};

    // Not a real token, and hyphenated so no secret scanner takes it for
    // one; the `hf_` prefix stays, so a filter that only matched the prefix
    // would still be caught.
    const SECRET: &str = "hf_example-token-for-these-tests-only";

    fn token() -> HfToken {
        HfToken::resolve(Ok(SECRET.to_string())).unwrap().unwrap()
    }

    fn authorization(client: &RegistryClient, url: &str) -> Option<HeaderValue> {
        let request = client.get(url).unwrap().build().expect("request builds");
        request.headers().get(AUTHORIZATION).cloned()
    }

    // ── Reading the variable ────────────────────────────────────────────

    #[test]
    fn unset_or_blank_is_no_token() {
        assert_eq!(HfToken::resolve(Err(VarError::NotPresent)), Ok(None));
        assert_eq!(HfToken::resolve(Ok(String::new())), Ok(None));
        assert_eq!(HfToken::resolve(Ok("  \n".to_string())), Ok(None));
    }

    // `HF_TOKEN=$(cat token-file)` keeps the file's newline.
    #[test]
    fn surrounding_whitespace_is_trimmed() {
        assert_eq!(
            HfToken::resolve(Ok(format!("  {SECRET}\n"))),
            Ok(Some(token()))
        );
    }

    #[test]
    fn a_value_no_header_can_carry_is_an_error_that_does_not_repeat_it() {
        let broken = [
            format!("{SECRET} extra"),
            format!("{}\n{}", &SECRET[..10], &SECRET[10..]),
            format!("{SECRET}é"),
        ];
        for raw in broken {
            let err = HfToken::resolve(Ok(raw)).unwrap_err();
            assert!(err.contains(HF_TOKEN_VAR), "{err}");
            assert!(
                !err.contains(&SECRET[..10]),
                "the error repeats the value: {err}"
            );
        }
        let not_unicode = std::ffi::OsString::from("x");
        assert!(HfToken::resolve(Err(VarError::NotUnicode(not_unicode))).is_err());
    }

    #[test]
    fn the_token_is_never_in_its_debug_output() {
        let shown = format!("{:?} {:?}", token(), Some(token()));
        assert!(!shown.contains(SECRET), "{shown}");
        assert!(!shown.contains(&SECRET[3..12]), "{shown}");
    }

    // ── Where it may go ─────────────────────────────────────────────────

    #[test]
    fn only_https_huggingface_co_on_the_default_port_is_hugging_face() {
        for url in [
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/q.gguf",
            "https://huggingface.co/api/models/Qwen/Qwen3-8B-GGUF",
            "https://HuggingFace.CO/api/models?search=qwen",
            "https://huggingface.co:443/api/models/o/r",
        ] {
            assert!(is_hf_url(url), "{url} is Hugging Face's");
        }
        for url in [
            "http://huggingface.co/o/r/resolve/main/q.gguf",
            "https://huggingface.co:8443/o/r/resolve/main/q.gguf",
            "https://hf.co/o/r/resolve/main/q.gguf",
            "https://cdn-lfs.huggingface.co/o/r/q.gguf",
            "https://us.aws.cdn.hf.co/xet-bridge-us/abc",
            "https://huggingface.co.example.com/o/r/resolve/main/q.gguf",
            "https://evilhuggingface.co/o/r/resolve/main/q.gguf",
            "https://huggingface.co./o/r/resolve/main/q.gguf",
            "https://huggingface.co@example.com/o/r/resolve/main/q.gguf",
            "https://example.com/huggingface.co/o/r/resolve/main/q.gguf",
            "https://example.com/?next=https://huggingface.co/o/r",
            "https://example.com/o/r/resolve/main/q.gguf",
            "file:///huggingface.co/o/r",
            "huggingface.co/o/r",
            "not a url",
        ] {
            assert!(!is_hf_url(url), "{url} must not get the token");
        }
    }

    // ── What each request carries ───────────────────────────────────────

    #[test]
    fn without_a_token_no_request_carries_authorization() {
        let client = RegistryClient::with_token(reqwest::Client::new(), None);
        for url in [
            "https://huggingface.co/o/r/resolve/main/q.gguf",
            "https://huggingface.co/api/models/o/r",
            "https://example.com/q.gguf",
        ] {
            assert_eq!(authorization(&client, url), None, "{url}");
            assert_eq!(client.token_use(url), TokenUse::Absent);
        }
    }

    #[test]
    fn with_a_token_hugging_face_requests_carry_it_as_a_sensitive_bearer() {
        let client = RegistryClient::with_token(reqwest::Client::new(), Some(token()));
        for url in [
            "https://huggingface.co/o/r/resolve/main/model-00001-of-00002.gguf",
            "https://huggingface.co/o/r/resolve/main/mmproj-F16.gguf",
            "https://huggingface.co/api/models/o/r",
            "https://huggingface.co/api/models/o/r/tree/main?recursive=1",
            "https://huggingface.co/api/models?search=q&filter=gguf",
        ] {
            let value = authorization(&client, url).unwrap_or_else(|| panic!("{url}: no token"));
            assert_eq!(value.to_str().unwrap(), format!("Bearer {SECRET}"));
            // Sensitive: printed as `Sensitive` by `Debug`, never the value.
            assert!(value.is_sensitive());
            assert!(!format!("{value:?}").contains(SECRET));
            assert_eq!(client.token_use(url), TokenUse::Sent);
        }
    }

    // The requirement in one test: a token configured for Hugging Face goes
    // nowhere else, not even to hosts that name Hugging Face somewhere.
    #[test]
    fn with_a_token_every_other_address_goes_without_it() {
        let client = RegistryClient::with_token(reqwest::Client::new(), Some(token()));
        for url in [
            "https://example.com/model.gguf",
            "https://hf.co/o/r/resolve/main/q.gguf",
            "http://huggingface.co/o/r/resolve/main/q.gguf",
            "https://evilhuggingface.co/o/r/resolve/main/q.gguf",
            "https://cdn-lfs.huggingface.co/o/r/q.gguf",
            "https://us.aws.cdn.hf.co/xet-bridge-us/abc",
            "https://huggingface.co.example.com/o/r/resolve/main/q.gguf",
            "https://example.com/?next=https://huggingface.co/o/r",
            "https://raw.githubusercontent.com/eullm/eullm/main/catalog.json",
        ] {
            assert_eq!(authorization(&client, url), None, "{url} got the token");
            assert_eq!(client.token_use(url), TokenUse::Withheld);
        }
    }

    // A malformed token fails what it was meant for, loudly, and nothing
    // else: a download from another address never needed it.
    #[test]
    fn an_unusable_token_fails_only_the_requests_it_would_have_gone_on() {
        let client = RegistryClient {
            http: reqwest::Client::new(),
            token: HfToken::resolve(Ok("hf_abc def".to_string())),
        };
        let err = client
            .get("https://huggingface.co/o/r/resolve/main/q.gguf")
            .unwrap_err();
        assert!(err.contains(HF_TOKEN_VAR), "{err}");
        let request = client
            .get("https://example.com/q.gguf")
            .expect("another address is unaffected")
            .build()
            .unwrap();
        assert_eq!(request.headers().get(AUTHORIZATION), None);
    }

    // The CDN hop of every download: `huggingface.co` answers with a
    // redirect to another host. What keeps the token off that host is
    // reqwest, not this module, so reqwest is held to it here — against
    // local servers, with a same-host redirect as the control that proves
    // the echo would have shown a header that got through.
    #[tokio::test]
    async fn reqwest_drops_the_token_on_a_redirect_to_another_host_or_port() {
        use axum::extract::Path;
        use axum::http::HeaderMap as Headers;
        use axum::response::Redirect;
        use axum::routing::get;
        use tokio::net::TcpListener;

        // What the server was sent, as the body.
        let echo = || {
            get(|headers: Headers| async move {
                headers
                    .get(AUTHORIZATION)
                    .map(|v| v.to_str().unwrap_or("").to_string())
                    .unwrap_or_default()
            })
        };
        let other_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let other = other_listener.local_addr().unwrap();
        let origin_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = origin_listener.local_addr().unwrap();

        let redirects = get(move |Path(target): Path<String>| async move {
            let to = match target.as_str() {
                // Same host, same port: the control.
                "same" => "/echo".to_string(),
                // Same host name, another port.
                "port" => format!("http://{other}/echo"),
                // The same server under another host name.
                _ => format!("http://localhost:{}/echo", origin.port()),
            };
            Redirect::temporary(&to)
        });
        let origin_app = axum::Router::new()
            .route("/echo", echo())
            .route("/to/{target}", redirects);
        let other_app = axum::Router::new().route("/echo", echo());
        tokio::spawn(async move {
            let _ = axum::serve(origin_listener, origin_app).await;
        });
        tokio::spawn(async move {
            let _ = axum::serve(other_listener, other_app).await;
        });

        // `no_proxy`: the test is about these two servers, and a proxy from
        // the environment would put a third one in between.
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let fetch = |target: &'static str| {
            let client = client.clone();
            async move {
                client
                    .get(format!("http://{origin}/to/{target}"))
                    .bearer_auth(SECRET)
                    .send()
                    .await
                    .unwrap()
                    .text()
                    .await
                    .unwrap()
            }
        };

        assert_eq!(fetch("same").await, format!("Bearer {SECRET}"), "control");
        assert_eq!(
            fetch("port").await,
            "",
            "another port must not see the token"
        );
        assert_eq!(
            fetch("host").await,
            "",
            "another host must not see the token"
        );
    }

    // ── Explaining a refusal ────────────────────────────────────────────

    const RESOLVE: &str = "https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/model.gguf";

    fn hf_headers(code: Option<&str>, message: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(code) = code {
            headers.insert("x-error-code", HeaderValue::from_str(code).unwrap());
        }
        headers.insert("x-error-message", HeaderValue::from_str(message).unwrap());
        headers
    }

    // What Hugging Face answered, checked live, to an anonymous download from
    // a gated repository.
    #[test]
    fn an_anonymous_gated_download_says_to_accept_the_terms_and_set_the_token() {
        let headers = hf_headers(
            Some("GatedRepo"),
            "Access to model meta-llama/Llama-3.2-1B is restricted. You must have access to it \
             and be authenticated to access it. Please log in.",
        );
        let msg = refusal_message(
            RESOLVE,
            StatusCode::UNAUTHORIZED,
            &headers,
            TokenUse::Absent,
        )
        .unwrap();
        assert!(
            msg.starts_with(
                "Hugging Face refused meta-llama/Llama-3.2-1B (HTTP 401: \"Access to model"
            ),
            "{msg}"
        );
        assert!(msg.contains("gated"), "{msg}");
        assert!(
            msg.contains("https://huggingface.co/meta-llama/Llama-3.2-1B"),
            "{msg}"
        );
        assert!(msg.contains("set HF_TOKEN"), "{msg}");
    }

    // A private repository and a missing one look the same from outside, and
    // Hugging Face answers both "Invalid username or password." — checked live.
    #[test]
    fn an_anonymous_refusal_without_a_gate_names_private_or_missing() {
        let headers = hf_headers(None, "Invalid username or password.");
        let msg = refusal_message(
            "https://huggingface.co/api/models/someone/private-repo",
            StatusCode::UNAUTHORIZED,
            &headers,
            TokenUse::Absent,
        )
        .unwrap();
        assert!(msg.contains("someone/private-repo"), "{msg}");
        assert!(msg.contains("private, or does not exist"), "{msg}");
        assert!(msg.contains("set HF_TOKEN"), "{msg}");
    }

    #[test]
    fn a_401_with_the_token_sent_blames_the_token() {
        let headers = hf_headers(Some("GatedRepo"), "Please log in.");
        let msg =
            refusal_message(RESOLVE, StatusCode::UNAUTHORIZED, &headers, TokenUse::Sent).unwrap();
        assert!(msg.contains("was not accepted"), "{msg}");
        assert!(!msg.contains("set HF_TOKEN"), "it is set: {msg}");
    }

    #[test]
    fn a_403_on_a_gated_repository_says_the_account_has_no_access_yet() {
        let headers = hf_headers(
            Some("GatedRepo"),
            "Access to model meta-llama/Llama-3.2-1B is restricted and you are not in the \
             authorized list.",
        );
        let msg =
            refusal_message(RESOLVE, StatusCode::FORBIDDEN, &headers, TokenUse::Sent).unwrap();
        assert!(msg.contains("not in the authorized list"), "{msg}");
        assert!(msg.contains("no access to it yet"), "{msg}");
        assert!(
            msg.contains("https://huggingface.co/meta-llama/Llama-3.2-1B"),
            "{msg}"
        );
    }

    #[test]
    fn a_403_without_a_gate_points_at_the_token_permissions() {
        let headers = hf_headers(None, "Forbidden");
        let msg =
            refusal_message(RESOLVE, StatusCode::FORBIDDEN, &headers, TokenUse::Sent).unwrap();
        assert!(msg.contains("fine-grained"), "{msg}");
    }

    // `hf.co` redirects to `huggingface.co`, and the token does not follow a
    // host change — so the refusal comes from Hugging Face while the token,
    // though set, never went. Telling that user to set it would be wrong.
    #[test]
    fn a_token_that_was_set_but_not_sent_is_named_as_such() {
        let headers = hf_headers(Some("GatedRepo"), "Please log in.");
        let msg = refusal_message(
            RESOLVE,
            StatusCode::UNAUTHORIZED,
            &headers,
            TokenUse::Withheld,
        )
        .unwrap();
        assert!(
            msg.contains("is set, but it is only sent to https://huggingface.co"),
            "{msg}"
        );
    }

    #[test]
    fn only_a_401_or_403_from_hugging_face_itself_is_explained() {
        let headers = hf_headers(Some("GatedRepo"), "no");
        for status in [
            StatusCode::NOT_FOUND,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::OK,
        ] {
            assert_eq!(
                refusal_message(RESOLVE, status, &headers, TokenUse::Sent),
                None
            );
        }
        // A CDN refusing an expired pre-signed link is not about the token.
        assert_eq!(
            refusal_message(
                "https://us.aws.cdn.hf.co/xet-bridge-us/abc?X-Amz-Signature=0",
                StatusCode::FORBIDDEN,
                &HeaderMap::new(),
                TokenUse::Sent,
            ),
            None
        );
    }

    // A header cannot hold a line break, but it can hold a tab, and it is
    // text from outside going into a terminal and a log line.
    #[test]
    fn hugging_faces_own_words_are_quoted_clean_and_short() {
        let long = format!("column one\tcolumn two {}", "x".repeat(1000));
        let headers = hf_headers(None, &long);
        let msg =
            refusal_message(RESOLVE, StatusCode::FORBIDDEN, &headers, TokenUse::Absent).unwrap();
        assert!(msg.contains("column onecolumn two"), "{msg}");
        assert!(!msg.contains('\t'), "{msg}");
        assert!(msg.len() < 800, "{} chars", msg.len());
    }

    #[test]
    fn the_repository_is_read_from_every_kind_of_hugging_face_url() {
        assert_eq!(repo_of(RESOLVE).as_deref(), Some("meta-llama/Llama-3.2-1B"));
        assert_eq!(
            repo_of("https://huggingface.co/api/models/o/r/tree/main?recursive=1").as_deref(),
            Some("o/r")
        );
        assert_eq!(
            repo_of("https://huggingface.co/api/resolve-cache/models/o/r/abc/q.gguf").as_deref(),
            Some("o/r")
        );
        assert_eq!(repo_of("https://huggingface.co/api/models?search=q"), None);
        assert_eq!(repo_of("https://huggingface.co/settings/tokens"), None);
    }
}
