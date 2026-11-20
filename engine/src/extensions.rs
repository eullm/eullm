//! What a program that embeds the engine can add to it.
//!
//! The engine is a library (`eullm_engine`) as well as the `eullm` binary,
//! so that a distribution can build its own binary around it: call
//! [`run_with_extensions`](crate::run_with_extensions) with an
//! [`ExtensionRegistry`] instead of `eullm_engine::main`, and everything
//! else, the command line included, is the engine's own. With the default
//! registry nothing changes: `eullm` itself is exactly that.
//!
//! The extension points are the two HTTP listeners. Each is handed over as
//! the finished [`axum::Router`], perimeter and all, and what a wrap returns
//! is served instead: it can add routes (`router.merge(...)`) and wrap every
//! request in its own middleware (`router.layer(...)`), which then runs
//! before the engine's API keys, address allowlist and origin checks. Two
//! things it can tell the engine about a request, by inserting them in the
//! request's extensions:
//!
//! - an [`Identity`] made with [`Identity::named`]: the engine takes it as
//!   authenticated, skips its own API keys for that request, and writes the
//!   name in the audit trail as the request's user;
//! - an [`AdminGrant`]: the request may do what the admin key allows (see
//!   `api::admin` and `api::managed`).
//!
//! Both are request extensions, which a client cannot send: only code
//! running in this process sets them. Routes a wrap adds are outside the
//! engine's perimeter, and are the wrap's to protect.

use std::sync::{Arc, OnceLock};

pub use crate::api::admin::AdminGrant;
pub use crate::api::auth::Identity;
/// The HTTP framework the routers are made with: an embedding program
/// builds its routes and middleware with this very version.
pub use axum;

/// Takes a listener's router and returns the one to serve.
pub type RouterWrap = Arc<dyn Fn(axum::Router) -> axum::Router + Send + Sync>;

/// What an embedding program adds. Every field defaults to nothing.
#[derive(Clone, Default)]
pub struct ExtensionRegistry {
    /// Wraps the chat listener's router (the UI, and the API beside it).
    pub chat: Option<RouterWrap>,
    /// Wraps the API listener's router (`/api`, `/v1`, `/metrics`).
    pub api: Option<RouterWrap>,
}

impl std::fmt::Debug for ExtensionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionRegistry")
            .field("chat", &self.chat.is_some())
            .field("api", &self.api.is_some())
            .finish()
    }
}

static INSTALLED: OnceLock<ExtensionRegistry> = OnceLock::new();

/// Install `registry` for this process. The first call wins; `false` when
/// one was installed already.
pub(crate) fn install(registry: ExtensionRegistry) -> bool {
    INSTALLED.set(registry).is_ok()
}

/// What is installed: the default (nothing) unless an embedding program
/// installed a registry before the server started.
pub(crate) fn installed() -> &'static ExtensionRegistry {
    INSTALLED.get_or_init(ExtensionRegistry::default)
}

/// `router` as the installed registry wants it served on the chat listener.
pub(crate) fn wrap_chat(router: axum::Router) -> axum::Router {
    match &installed().chat {
        Some(wrap) => wrap(router),
        None => router,
    }
}

/// `router` as the installed registry wants it served on the API listener.
pub(crate) fn wrap_api(router: axum::Router) -> axum::Router {
    match &installed().api {
        Some(wrap) => wrap(router),
        None => router,
    }
}

/// The engine's version and the commit it was built from, for an embedding
/// program's own `--version`: read from the engine it was compiled with, so
/// they cannot drift from what the binary really holds.
pub fn build_info() -> (&'static str, &'static str) {
    (env!("CARGO_PKG_VERSION"), env!("EULLM_GIT_HASH"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_registry_serves_the_router_as_it_is() {
        let registry = ExtensionRegistry::default();
        assert!(registry.chat.is_none() && registry.api.is_none());
        assert_eq!(
            format!("{registry:?}"),
            "ExtensionRegistry { chat: false, api: false }"
        );
    }

    #[test]
    fn a_named_identity_counts_as_authenticated() {
        let who = Identity::named("mario.rossi");
        assert!(who.is_authenticated());
        assert_eq!(who.key_id(), Some("mario.rossi"));
    }
}
