//! Local model management — storage, manifests, and catalog.

pub mod catalog;
pub mod store;

pub use catalog::{CatalogEntry, EU_CATALOG};
pub use store::ModelStore;
