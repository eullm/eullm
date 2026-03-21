//! Ollama-compatible REST API.
//!
//! Exposes endpoints compatible with the Ollama API specification
//! so that existing tools (Open WebUI, LangChain, n8n) work out of the box.

mod routes;

use axum::Router;
use tokio::net::TcpListener;

/// Start the API server on the given port.
pub async fn serve(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app = router();
    let addr = format!("0.0.0.0:{port}");
    tracing::info!("EULLM Engine listening on {addr}");

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Build the Ollama-compatible API router.
fn router() -> Router {
    Router::new()
        .nest("/api", routes::api_routes())
        .nest("/v1", routes::openai_routes())
}
