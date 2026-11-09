//! A safe wrapper around `llama_model`.
use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_int;
use std::path::Path;
use std::ptr;
use std::str::Utf8Error;

use crate::context::params::LlamaContextParams;
use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::model::params::LlamaModelParams;
use crate::ptr::Ptr;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;
use crate::vocab::LlamaVocab;
use crate::{
    ApplyChatTemplateError, ChatTemplateError, JinjaChatTemplateError, LlamaContextLoadError,
    LlamaLoraAdapterInitError, LlamaModelLoadError, MetaValError, NewLlamaChatMessageError,
};

pub mod params;
// For backwards compat.
pub use crate::vocab::{LlamaTokenTypeFromIntError, VocabType};

/// A safe wrapper around `llama_model`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaModel {
    pub(crate) model: Ptr<llama_cpp_sys_2::llama_model>,
}

// SAFETY: The model is immutable except for methods where it's passed as `&mut`.
//
// FIXME(madsmtm): Apart from `llama_new_context_with_model`, which mutates.
unsafe impl Send for LlamaModel {}
// SAFETY: Same as above.
unsafe impl Sync for LlamaModel {}

/// A safe wrapper around `llama_lora_adapter`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaLoraAdapter {
    pub(crate) lora_adapter: Ptr<llama_cpp_sys_2::llama_adapter_lora>,
}

// SAFETY: The lora is immutable.
unsafe impl Send for LlamaLoraAdapter {}
// SAFETY: Same as above.
unsafe impl Sync for LlamaLoraAdapter {}

/// A performance-friendly wrapper around [`LlamaModel::chat_template`] which is then
/// fed into [`LlamaModel::apply_chat_template`] to convert a list of messages into an LLM
/// prompt. Internally the template is stored as a `CString` to avoid round-trip conversions
/// within the FFI.
#[derive(Eq, PartialEq, Clone, PartialOrd, Ord, Hash)]
pub struct LlamaChatTemplate(CString);

impl LlamaChatTemplate {
    /// Create a new template from a string. This can either be the name of a llama.cpp [chat template](https://github.com/ggerganov/llama.cpp/blob/8a8c4ceb6050bd9392609114ca56ae6d26f5b8f5/src/llama-chat.cpp#L27-L61)
    /// like "chatml" or "llama3" or an actual Jinja template for llama.cpp to interpret.
    pub fn new(template: &str) -> Result<Self, std::ffi::NulError> {
        Ok(Self(CString::new(template)?))
    }

    /// Accesses the template as a c string reference.
    pub fn as_c_str(&self) -> &CStr {
        &self.0
    }

    /// Attempts to convert the `CString` into a Rust str reference.
    pub fn to_str(&self) -> Result<&str, Utf8Error> {
        self.0.to_str()
    }

    /// Convenience method to create an owned String.
    pub fn to_string(&self) -> Result<String, Utf8Error> {
        self.to_str().map(str::to_string)
    }
}

impl std::fmt::Debug for LlamaChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A Safe wrapper around `llama_chat_message`
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatMessage {
    role: CString,
    content: CString,
}

impl LlamaChatMessage {
    /// Create a new `LlamaChatMessage`
    ///
    /// # Errors
    /// If either of ``role`` or ``content`` contain null bytes.
    pub fn new(role: String, content: String) -> Result<Self, NewLlamaChatMessageError> {
        Ok(Self {
            role: CString::new(role)?,
            content: CString::new(content)?,
        })
    }
}

/// Result of rendering a model's own chat template through llama.cpp's Jinja
/// engine (`common_chat_templates_apply`), returned by
/// [`LlamaModel::apply_jinja_chat_template`].
#[derive(Debug, Clone)]
pub struct JinjaChatTemplateResult {
    /// The rendered prompt, ready to tokenize.
    pub prompt: String,
    /// `true` only when the GGUF actually carried its own chat template.
    /// When `false`, llama.cpp silently fell back to a built-in ChatML
    /// template internally, and `prompt` reflects that fallback rather than
    /// anything specific to this model — callers should prefer a
    /// known-good template of their own in that case.
    pub was_explicit: bool,
    /// The opening delimiter of a reasoning/thinking block, when the
    /// template declares one (e.g. `<think>`, or `<|channel|>thought\n`).
    pub thinking_start_tag: Option<String>,
    /// The closing delimiter matching `thinking_start_tag`. When a template
    /// declares more than one closing form, this is only the first.
    pub thinking_end_tag: Option<String>,
}

/// Result of rendering a chat template from OpenAI-format request JSON,
/// returned by [`LlamaModel::apply_chat_template_oaicompat`]. Besides the
/// prompt it carries the output-format descriptor ([`Self::format`],
/// [`Self::parser`], [`Self::generation_prompt`]) that [`chat_parse`] needs
/// to turn the model's raw output back into structured content /
/// reasoning_content / tool_calls.
#[derive(Debug, Clone)]
pub struct OaiChatTemplateResult {
    /// The rendered prompt, ready to tokenize.
    pub prompt: String,
    /// See [`JinjaChatTemplateResult::was_explicit`].
    pub was_explicit: bool,
    /// llama.cpp's `common_chat_format` discriminant for the output syntax
    /// this prompt sets the model up to produce.
    pub format: i32,
    /// The parser of that output syntax. Opaque; pass to [`chat_parse`].
    pub parser: Option<ChatParser>,
    /// The generation prompt suffix, used by the parser to recognize
    /// message boundaries. May be empty.
    pub generation_prompt: String,
    /// See [`JinjaChatTemplateResult::thinking_start_tag`].
    pub thinking_start_tag: Option<String>,
    /// See [`JinjaChatTemplateResult::thinking_end_tag`].
    pub thinking_end_tag: Option<String>,
}

/// The parser of the output format a chat prompt sets the model up to
/// produce, from [`LlamaModel::apply_chat_template_oaicompat`]. EuLLM: since
/// llama.cpp #30210 it is a C++ object with no saved form, so it is held as a
/// handle, freed with its last clone. llama.cpp only reads it to parse, so it
/// may be shared between threads.
#[derive(Clone)]
pub struct ChatParser(std::sync::Arc<ChatParserHandle>);

struct ChatParserHandle(ptr::NonNull<llama_cpp_sys_2::llama_rs_chat_parser>);

impl Drop for ChatParserHandle {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_rs_chat_parser_free(self.0.as_ptr()) }
    }
}

// SAFETY: the parser is created whole before the handle exists and is only
// read afterwards (`common_chat_parse` takes it by const reference); it is
// freed once, by the last clone's drop.
unsafe impl Send for ChatParserHandle {}
unsafe impl Sync for ChatParserHandle {}

impl std::fmt::Debug for ChatParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ChatParser")
    }
}

/// Parse a model's raw output with the parser from
/// [`LlamaModel::apply_chat_template_oaicompat`], or as plain content when
/// `parser` is `None`. Model-free — callable from any thread, at any time
/// after the render. Returns the parsed message as OpenAI-compatible JSON
/// (`{"role","content","reasoning_content","tool_calls",...}`).
///
/// # Errors
/// See [`JinjaChatTemplateError`].
pub fn chat_parse(
    input: &str,
    is_partial: bool,
    parser: Option<&ChatParser>,
) -> Result<String, JinjaChatTemplateError> {
    let input_c = CString::new(input)?;
    let mut out_json: *mut c_char = ptr::null_mut();

    let status = unsafe {
        llama_cpp_sys_2::llama_rs_chat_parse(
            input_c.as_ptr(),
            is_partial,
            parser.map_or(ptr::null(), |p| p.0 .0.as_ptr().cast_const()),
            &raw mut out_json,
        )
    };

    match status {
        llama_cpp_sys_2::LLAMA_RS_STATUS_OK => {
            if out_json.is_null() {
                return Err(JinjaChatTemplateError::NullResult);
            }
            let bytes = unsafe { CStr::from_ptr(out_json) }.to_bytes().to_vec();
            unsafe {
                llama_cpp_sys_2::llama_rs_string_free(out_json);
            }
            Ok(String::from_utf8(bytes)?)
        }
        llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT => {
            Err(JinjaChatTemplateError::InvalidArgument)
        }
        llama_cpp_sys_2::LLAMA_RS_STATUS_ALLOCATION_FAILED => {
            Err(JinjaChatTemplateError::AllocationFailed)
        }
        llama_cpp_sys_2::LLAMA_RS_STATUS_EXCEPTION => Err(JinjaChatTemplateError::Exception),
        other => Err(JinjaChatTemplateError::UnknownStatus(other)),
    }
}

/// The Rope type that's used within the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    Norm,
    NeoX,
    MRope,
    Vision,
}

impl LlamaModel {
    /// Get the model's vocabulary.
    #[must_use]
    pub fn vocab(&self) -> LlamaVocab<'_> {
        let ptr = unsafe { llama_cpp_sys_2::llama_model_get_vocab(self.model.as_ptr()) };
        LlamaVocab::new(ptr).expect("model must have vocabulary")
    }

    /// get the number of tokens the model was trained on
    ///
    /// # Panics
    ///
    /// If the number of tokens the model was trained on does not fit into an `u32`. This should be impossible on most
    /// platforms due to llama.cpp returning a `c_int` (i32 on most platforms) which is almost certainly positive.
    #[must_use]
    pub fn n_ctx_train(&self) -> u32 {
        let n_ctx_train = unsafe { llama_cpp_sys_2::llama_n_ctx_train(self.model.as_ptr()) };
        u32::try_from(n_ctx_train).expect("n_ctx_train fits into an u32")
    }

    /// Get all tokens in the model.
    pub fn tokens(&self, decode_special: bool) -> impl Iterator<Item = (LlamaToken, Vec<u8>)> + '_ {
        let vocab = self.vocab();
        vocab.tokens().map(move |llama_token| {
            let bytes = vocab.token_to_piece(llama_token, decode_special, None);
            (llama_token, bytes)
        })
    }

    /// Get the decoder start token.
    #[must_use]
    pub fn decode_start_token(&self) -> LlamaToken {
        let token =
            unsafe { llama_cpp_sys_2::llama_model_decoder_start_token(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// The number of tokens the model was trained on.
    ///
    /// This returns a `c_int` for maximum compatibility. Most of the time it can be cast to an i32
    /// without issue.
    #[must_use]
    pub fn n_vocab(&self) -> i32 {
        self.vocab().n_tokens()
    }

    /// This returns a `c_int` for maximum compatibility. Most of the time it can be cast to an i32
    /// without issue.
    #[must_use]
    pub fn n_embd(&self) -> c_int {
        unsafe { llama_cpp_sys_2::llama_n_embd(self.model.as_ptr()) }
    }

    /// The model's *output* embedding width (`n_embd_out`). This is the width
    /// llama.cpp actually extracts embeddings at — `n_embd` and `n_embd_out`
    /// diverge when `{arch}.embedding_length_out` is present (deepstack models
    /// like qwen3vl). Returns a `c_int` for maximum compatibility.
    #[must_use]
    pub fn n_embd_out(&self) -> c_int {
        unsafe { llama_cpp_sys_2::llama_model_n_embd_out(self.model.as_ptr()) }
    }

    /// The model's classification output width (`n_cls_out`, default 1) — the
    /// width of a RANK-pooled embeddings read (llama.h:1029).
    #[must_use]
    pub fn n_cls_out(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_model_n_cls_out(self.model.as_ptr()) }
    }

    /// Returns the total size of all the tensors in the model in bytes.
    pub fn size(&self) -> u64 {
        unsafe { llama_cpp_sys_2::llama_model_size(self.model.as_ptr()) }
    }

    /// Returns the number of parameters in the model.
    pub fn n_params(&self) -> u64 {
        unsafe { llama_cpp_sys_2::llama_model_n_params(self.model.as_ptr()) }
    }

    /// Write the model's current weights and metadata to a GGUF file — after
    /// training, the trained weights.
    ///
    /// llama.cpp's saver reports no error, so this removes whatever is at
    /// `path` first and checks that a file is there afterwards. It writes
    /// the metadata as the model holds it *now*: training
    /// (`crate::opt::Trainer`) sets the context length to the one trained at,
    /// and callers that want the original back have to patch the file.
    ///
    /// # Errors
    ///
    /// A path that is not UTF-8 or contains a NUL, one whose old file cannot
    /// be removed, or no file afterwards.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let path = path.as_ref();
        let c_path = path
            .to_str()
            .and_then(|s| CString::new(s).ok())
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "path is not valid UTF-8")
            })?;
        match std::fs::remove_file(path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
        unsafe { llama_cpp_sys_2::llama_model_save_to_file(self.model.as_ptr(), c_path.as_ptr()) };
        if path.is_file() {
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("llama.cpp wrote no file at {}", path.display()),
            ))
        }
    }

    /// Returns whether the model is a recurrent network (Mamba, RWKV, etc)
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_model_is_recurrent(self.model.as_ptr()) }
    }

    /// Returns whether the model is a hybrid network (Jamba, Granite, Qwen3xx, etc)
    ///
    /// Hybrid models have both attention layers and recurrent/SSM layers.
    /// They require special handling for state checkpointing.
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_model_is_hybrid(self.model.as_ptr()) }
    }

    /// EuLLM addition: the number of multi-token prediction (MTP, "nextn")
    /// layers the model was loaded with — 0 for a model without them, or one
    /// loaded without `load_mtp` (see `LlamaModelParams::with_load_mtp`).
    #[must_use]
    pub fn n_layer_nextn(&self) -> u32 {
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_layer_nextn(self.model.as_ptr()) })
            .unwrap_or(0)
    }

    /// Returns the number of layers within the model.
    pub fn n_layer(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_layer(self.model.as_ptr()) }).unwrap()
    }

    /// Returns the number of attention heads within the model.
    pub fn n_head(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_head(self.model.as_ptr()) }).unwrap()
    }

    /// Returns the number of KV attention heads.
    pub fn n_head_kv(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_head_kv(self.model.as_ptr()) })
            .unwrap()
    }

    /// Get metadata value as a string by key name
    pub fn meta_val_str(&self, key: &str) -> Result<String, MetaValError> {
        let key_cstring = CString::new(key)?;
        let key_ptr = key_cstring.as_ptr();

        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_sys_2::llama_model_meta_val_str(
                    self.model.as_ptr(),
                    key_ptr,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// Get the number of metadata key/value pairs
    pub fn meta_count(&self) -> i32 {
        unsafe { llama_cpp_sys_2::llama_model_meta_count(self.model.as_ptr()) }
    }

    /// Get metadata key name by index
    pub fn meta_key_by_index(&self, index: i32) -> Result<String, MetaValError> {
        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_sys_2::llama_model_meta_key_by_index(
                    self.model.as_ptr(),
                    index,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// Get metadata value as a string by index
    pub fn meta_val_str_by_index(&self, index: i32) -> Result<String, MetaValError> {
        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_sys_2::llama_model_meta_val_str_by_index(
                    self.model.as_ptr(),
                    index,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// Returns the rope type of the model.
    pub fn rope_type(&self) -> Option<RopeType> {
        match unsafe { llama_cpp_sys_2::llama_model_rope_type(self.model.as_ptr()) } {
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_NONE => None,
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_NORM => Some(RopeType::Norm),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_NEOX => Some(RopeType::NeoX),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_MROPE => Some(RopeType::MRope),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_VISION => Some(RopeType::Vision),
            rope_type => {
                tracing::error!(rope_type = rope_type, "Unexpected rope type from llama.cpp");
                None
            }
        }
    }

    /// Get chat template from model by name. If the name parameter is None, the default chat template will be returned.
    ///
    /// You supply this into [`Self::apply_chat_template`] to get back a string with the appropriate template
    /// substitution applied to convert a list of messages into a prompt the LLM can use to complete
    /// the chat.
    ///
    /// You could also use an external jinja parser, like [minijinja](https://github.com/mitsuhiko/minijinja),
    /// to parse jinja templates not supported by the llama.cpp template engine.
    ///
    /// # Errors
    ///
    /// * If the model has no chat template by that name
    /// * If the chat template is not a valid [`CString`].
    pub fn chat_template(
        &self,
        name: Option<&str>,
    ) -> Result<LlamaChatTemplate, ChatTemplateError> {
        let name_cstr = name.map(CString::new);
        let name_ptr = match name_cstr {
            Some(Ok(name)) => name.as_ptr(),
            _ => std::ptr::null(),
        };
        let result =
            unsafe { llama_cpp_sys_2::llama_model_chat_template(self.model.as_ptr(), name_ptr) };

        // Convert result to Rust String if not null
        if result.is_null() {
            Err(ChatTemplateError::MissingTemplate)
        } else {
            let chat_template_cstr = unsafe { CStr::from_ptr(result) };
            let chat_template = CString::new(chat_template_cstr.to_bytes())?;
            Ok(LlamaChatTemplate(chat_template))
        }
    }

    /// Loads a model from a file.
    ///
    /// # Errors
    ///
    /// See [`LlamaModelLoadError`] for more information.
    #[tracing::instrument(skip_all, fields(params))]
    pub fn load_from_file(
        _: &LlamaBackend,
        path: impl AsRef<Path>,
        params: &LlamaModelParams,
    ) -> Result<Self, LlamaModelLoadError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");
        let path = path
            .to_str()
            .ok_or(LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let llama_model =
            unsafe { llama_cpp_sys_2::llama_load_model_from_file(cstr.as_ptr(), params.params) };

        let model = Ptr::new(llama_model).ok_or(LlamaModelLoadError::NullResult)?;

        tracing::debug!(?path, "Loaded model");
        Ok(LlamaModel { model })
    }

    /// Initializes a lora adapter from a file.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterInitError`] for more information.
    pub fn lora_adapter_init(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<LlamaLoraAdapter, LlamaLoraAdapterInitError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");

        let path = path
            .to_str()
            .ok_or(LlamaLoraAdapterInitError::PathToStrError(
                path.to_path_buf(),
            ))?;

        let cstr = CString::new(path)?;
        let adapter = unsafe {
            llama_cpp_sys_2::llama_adapter_lora_init(self.model.as_mut_ptr(), cstr.as_ptr())
        };

        let adapter = Ptr::new(adapter).ok_or(LlamaLoraAdapterInitError::NullResult)?;

        tracing::debug!(?path, "Initialized lora adapter");
        Ok(LlamaLoraAdapter {
            lora_adapter: adapter,
        })
    }

    /// Create a new context from this model.
    ///
    /// # Errors
    ///
    /// There is many ways this can fail. See [`LlamaContextLoadError`] for more information.
    // we intentionally do not derive Copy on `LlamaContextParams` to allow llama.cpp to change the type to be non-trivially copyable.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_context<'a>(
        &'a self,
        _: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<LlamaContext<'a>, LlamaContextLoadError> {
        let context_params = params.context_params;
        // Constructing multiple contexts from a model in parallel is actually
        // unsound, since `llama_new_context_with_model` mutates the model's
        // `n_ctx_train` in some cases:
        // <https://github.com/ggml-org/llama.cpp/blob/f45576aa86c07d60d346b469651a098a15cc4f81/src/llama-context.cpp#L3751>
        // FIXME(madsmtm): Make this sound!
        let context = unsafe {
            llama_cpp_sys_2::llama_new_context_with_model(
                self.model.as_mut_ptr_unsound(),
                context_params,
            )
        };
        let context = Ptr::new(context).ok_or(LlamaContextLoadError::NullReturn)?;

        Ok(LlamaContext::new(context))
    }

    /// Create a new context bound to another context via llama.cpp's `ctx_other` field.
    ///
    /// This is required for MTP speculative decoding when the target model's
    /// architecture uses `LLM_ARCH_GEMMA4_ASSISTANT`, which asserts that the draft
    /// context references the target context so KV state can be shared.
    ///
    /// # Errors
    ///
    /// See [`LlamaContextLoadError`].
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_context_with_ctx_other<'a>(
        &'a self,
        _: &LlamaBackend,
        params: LlamaContextParams,
        ctx_other: &LlamaContext<'_>,
    ) -> Result<LlamaContext<'a>, LlamaContextLoadError> {
        let mut context_params = params.context_params;
        // FIXME(madsmtm): Use `.as_ptr()` after:
        // https://github.com/ggml-org/llama.cpp/pull/28316
        context_params.ctx_other = unsafe { ctx_other.context.as_mut_ptr_unsound() };
        // Unsoundness: See `LlamaMode::new_context`.
        let context = unsafe {
            llama_cpp_sys_2::llama_new_context_with_model(
                self.model.as_mut_ptr_unsound(),
                context_params,
            )
        };
        let context = Ptr::new(context).ok_or(LlamaContextLoadError::NullReturn)?;

        Ok(LlamaContext::new(context))
    }

    /// Creates a new context with backend samplers attached for specific sequences.
    ///
    /// Ownership of the samplers is transferred to the context, ensuring they remain
    /// alive for the context's lifetime. Only samplers that support backend execution
    /// (greedy, dist, temp, top_k, top_p, min_p, logit_bias) will run on the backend.
    ///
    /// # Arguments
    ///
    /// * `params` - Context parameters
    /// * `samplers` - Iterator of `(seq_id, sampler)` pairs where sampler must be a chain
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sampler = LlamaSampler::chain([
    ///     LlamaSampler::min_p(0.01, 64),
    ///     LlamaSampler::temp(0.1),
    ///     LlamaSampler::dist(42),
    /// ], false);
    ///
    /// let ctx = model.new_context_with_samplers(
    ///     &backend,
    ///     ctx_params,
    ///     [(0, sampler)],
    /// )?;
    /// ```
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_context_with_samplers<'a>(
        &'a self,
        _: &LlamaBackend,
        params: LlamaContextParams,
        samplers: impl IntoIterator<Item = (i32, LlamaSampler)>,
    ) -> Result<LlamaContext<'a>, LlamaContextLoadError> {
        let mut samplers: Vec<_> = samplers.into_iter().collect();
        let mut context_params = params.context_params;

        let mut sampler_configs: Vec<llama_cpp_sys_2::llama_sampler_seq_config> = samplers
            .iter_mut()
            .map(
                |(seq_id, sampler)| llama_cpp_sys_2::llama_sampler_seq_config {
                    seq_id: *seq_id,
                    sampler: sampler.sampler.as_mut_ptr(),
                },
            )
            .collect();

        if !sampler_configs.is_empty() {
            context_params.samplers = sampler_configs.as_mut_ptr();
            context_params.n_samplers = sampler_configs.len();
        }

        // Unsoundness: See `LlamaMode::new_context`.
        let context = unsafe {
            llama_cpp_sys_2::llama_new_context_with_model(
                self.model.as_mut_ptr_unsound(),
                context_params,
            )
        };
        let context = Ptr::new(context).ok_or(LlamaContextLoadError::NullReturn)?;

        Ok(LlamaContext::with_samplers(context, samplers))
    }

    /// Apply the models chat template to some messages.
    /// See <https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template>
    ///
    /// Unlike the llama.cpp `apply_chat_template` which just randomly uses the ChatML template when given
    /// a null pointer for the template, this requires an explicit template to be specified. If you want to
    /// use "chatml", then just do `LlamaChatTemplate::new("chatml")` or any other model name or template
    /// string.
    ///
    /// Use [`Self::chat_template`] to retrieve the template baked into the model (this is the preferred
    /// mechanism as using the wrong chat template can result in really unexpected responses from the LLM).
    ///
    /// You probably want to set `add_ass` to true so that the generated template string ends with a the
    /// opening tag of the assistant. If you fail to leave a hanging chat tag, the model will likely generate
    /// one into the output and the output may also have unexpected output aside from that.
    ///
    /// # Errors
    /// There are many ways this can fail. See [`ApplyChatTemplateError`] for more information.
    #[tracing::instrument(skip_all)]
    pub fn apply_chat_template(
        &self,
        tmpl: &LlamaChatTemplate,
        chat: &[LlamaChatMessage],
        add_ass: bool,
    ) -> Result<String, ApplyChatTemplateError> {
        // Buffer is twice the length of messages per their recommendation
        let message_length = chat.iter().fold(0, |acc, c| {
            acc + c.role.to_bytes().len() + c.content.to_bytes().len()
        });
        let mut buff: Vec<u8> = vec![0; message_length * 2];

        // Build our llama_cpp_sys_2 chat messages
        let chat: Vec<llama_cpp_sys_2::llama_chat_message> = chat
            .iter()
            .map(|c| llama_cpp_sys_2::llama_chat_message {
                role: c.role.as_ptr(),
                content: c.content.as_ptr(),
            })
            .collect();

        let tmpl_ptr = tmpl.0.as_ptr();

        let res = unsafe {
            llama_cpp_sys_2::llama_chat_apply_template(
                tmpl_ptr,
                chat.as_ptr(),
                chat.len(),
                add_ass,
                buff.as_mut_ptr().cast::<c_char>(),
                buff.len().try_into().expect("Buffer size exceeds i32::MAX"),
            )
        };

        if res < 0 {
            return Err(ApplyChatTemplateError::FfiError(res));
        }

        if res > buff.len().try_into().expect("Buffer size exceeds i32::MAX") {
            buff.resize(res.try_into().expect("res is negative"), 0);

            let res = unsafe {
                llama_cpp_sys_2::llama_chat_apply_template(
                    tmpl_ptr,
                    chat.as_ptr(),
                    chat.len(),
                    add_ass,
                    buff.as_mut_ptr().cast::<c_char>(),
                    buff.len().try_into().expect("Buffer size exceeds i32::MAX"),
                )
            };
            if res < 0 {
                return Err(ApplyChatTemplateError::FfiError(res));
            }
            assert_eq!(Ok(res), buff.len().try_into());
        }
        buff.truncate(res.try_into().expect("res is negative"));
        Ok(String::from_utf8(buff)?)
    }

    /// Render `messages` through this model's own Jinja chat template — the
    /// template embedded in the GGUF, applied with llama.cpp's own Jinja
    /// engine (`common_chat_templates_apply`, `minja` under the hood) — the
    /// same mechanism llama-server uses by default.
    ///
    /// Unlike [`Self::apply_chat_template`], which only supports llama.cpp's
    /// fixed list of pre-defined formats and cannot parse arbitrary Jinja,
    /// this renders whatever template the model actually ships with,
    /// including custom ones (tool-calling formats, reasoning-channel
    /// formats, and anything else `common_chat_templates_apply` supports).
    ///
    /// Check [`JinjaChatTemplateResult::was_explicit`] before trusting the
    /// result: when `false`, the GGUF had no embedded template at all and
    /// llama.cpp rendered a built-in ChatML fallback instead — callers with
    /// their own known-good template for this situation should prefer it.
    ///
    /// Text-only: each message is a single content string, not the
    /// structured content-parts some templates use for multimodal input.
    ///
    /// `enable_thinking` maps to llama.cpp's
    /// `common_chat_templates_inputs.enable_thinking`: templates with a
    /// reasoning toggle (Qwen3 family) render their own suppression form
    /// when `false`; templates without one ignore it.
    ///
    /// # Errors
    /// See [`JinjaChatTemplateError`].
    pub fn apply_jinja_chat_template(
        &self,
        messages: &[LlamaChatMessage],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<JinjaChatTemplateResult, JinjaChatTemplateError> {
        let role_ptrs: Vec<*const c_char> = messages.iter().map(|m| m.role.as_ptr()).collect();
        let content_ptrs: Vec<*const c_char> = messages.iter().map(|m| m.content.as_ptr()).collect();

        let mut was_explicit = false;
        let mut out_prompt: *mut c_char = ptr::null_mut();
        let mut out_thinking_start: *mut c_char = ptr::null_mut();
        let mut out_thinking_end: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_sys_2::llama_rs_apply_chat_template(
                self.model.as_ptr(),
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len(),
                add_generation_prompt,
                enable_thinking,
                &raw mut was_explicit,
                &raw mut out_prompt,
                &raw mut out_thinking_start,
                &raw mut out_thinking_end,
            )
        };

        // Takes ownership of an optional C string allocated by the wrapper
        // (freeing it via `llama_rs_string_free`), converting it to an owned
        // `String`. Returns `Ok(None)` for a null pointer, which the wrapper
        // uses for "the template declares no such tag" — not an error.
        let take_optional_string = |raw: *mut c_char| -> Result<Option<String>, JinjaChatTemplateError> {
            if raw.is_null() {
                return Ok(None);
            }
            let bytes = unsafe { CStr::from_ptr(raw) }.to_bytes().to_vec();
            unsafe {
                llama_cpp_sys_2::llama_rs_string_free(raw);
            }
            Ok(Some(String::from_utf8(bytes)?))
        };

        match status {
            llama_cpp_sys_2::LLAMA_RS_STATUS_OK => {
                let prompt = take_optional_string(out_prompt)?
                    .ok_or(JinjaChatTemplateError::NullResult)?;
                let thinking_start_tag = take_optional_string(out_thinking_start)?;
                let thinking_end_tag = take_optional_string(out_thinking_end)?;
                Ok(JinjaChatTemplateResult {
                    prompt,
                    was_explicit,
                    thinking_start_tag,
                    thinking_end_tag,
                })
            }
            llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT => {
                Err(JinjaChatTemplateError::InvalidArgument)
            }
            llama_cpp_sys_2::LLAMA_RS_STATUS_ALLOCATION_FAILED => {
                Err(JinjaChatTemplateError::AllocationFailed)
            }
            llama_cpp_sys_2::LLAMA_RS_STATUS_EXCEPTION => Err(JinjaChatTemplateError::Exception),
            other => Err(JinjaChatTemplateError::UnknownStatus(other)),
        }
    }

    /// Render a chat template from OpenAI-format request JSON — messages as
    /// the request's own `messages` array (tool roles and assistant
    /// `tool_calls` in history survive, unlike the `(role, content)`
    /// projection of [`Self::apply_jinja_chat_template`]) plus the optional
    /// `tools` array and `tool_choice`. The result carries the
    /// output-format descriptor that [`chat_parse`] consumes.
    ///
    /// # Errors
    /// See [`JinjaChatTemplateError`].
    pub fn apply_chat_template_oaicompat(
        &self,
        messages_json: &str,
        tools_json: Option<&str>,
        tool_choice: Option<&str>,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<OaiChatTemplateResult, JinjaChatTemplateError> {
        let messages_c = CString::new(messages_json)?;
        let tools_c = tools_json.map(CString::new).transpose()?;
        let tool_choice_c = tool_choice.map(CString::new).transpose()?;

        let mut was_explicit = false;
        let mut out_prompt: *mut c_char = ptr::null_mut();
        let mut out_format: i32 = 0;
        let mut out_parser: *mut llama_cpp_sys_2::llama_rs_chat_parser = ptr::null_mut();
        let mut out_generation_prompt: *mut c_char = ptr::null_mut();
        let mut out_thinking_start: *mut c_char = ptr::null_mut();
        let mut out_thinking_end: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_sys_2::llama_rs_apply_chat_template_oai(
                self.model.as_ptr(),
                messages_c.as_ptr(),
                tools_c.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
                tool_choice_c.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
                add_generation_prompt,
                enable_thinking,
                &raw mut was_explicit,
                &raw mut out_prompt,
                &raw mut out_format,
                &raw mut out_parser,
                &raw mut out_generation_prompt,
                &raw mut out_thinking_start,
                &raw mut out_thinking_end,
            )
        };
        // Owned from here, so every return below frees it.
        let parser =
            ptr::NonNull::new(out_parser).map(|p| ChatParser(std::sync::Arc::new(ChatParserHandle(p))));

        let take_optional_string = |raw: *mut c_char| -> Result<Option<String>, JinjaChatTemplateError> {
            if raw.is_null() {
                return Ok(None);
            }
            let bytes = unsafe { CStr::from_ptr(raw) }.to_bytes().to_vec();
            unsafe {
                llama_cpp_sys_2::llama_rs_string_free(raw);
            }
            Ok(Some(String::from_utf8(bytes)?))
        };

        match status {
            llama_cpp_sys_2::LLAMA_RS_STATUS_OK => {
                let prompt = take_optional_string(out_prompt)?
                    .ok_or(JinjaChatTemplateError::NullResult)?;
                let generation_prompt =
                    take_optional_string(out_generation_prompt)?.unwrap_or_default();
                let thinking_start_tag = take_optional_string(out_thinking_start)?;
                let thinking_end_tag = take_optional_string(out_thinking_end)?;
                Ok(OaiChatTemplateResult {
                    prompt,
                    was_explicit,
                    format: out_format,
                    parser,
                    generation_prompt,
                    thinking_start_tag,
                    thinking_end_tag,
                })
            }
            llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT => {
                Err(JinjaChatTemplateError::InvalidArgument)
            }
            llama_cpp_sys_2::LLAMA_RS_STATUS_ALLOCATION_FAILED => {
                Err(JinjaChatTemplateError::AllocationFailed)
            }
            llama_cpp_sys_2::LLAMA_RS_STATUS_EXCEPTION => Err(JinjaChatTemplateError::Exception),
            other => Err(JinjaChatTemplateError::UnknownStatus(other)),
        }
    }
}

/// Generic helper function for extracting string values from the C API
/// This are specifically useful for the the metadata functions, where we pass in a buffer
/// to be populated by a string, not yet knowing if the buffer is large enough.
/// If the buffer was not large enough, we get the correct length back, which can be used to
/// construct a buffer of appropriate size.
fn extract_meta_string<F>(c_function: F, capacity: usize) -> Result<String, MetaValError>
where
    F: Fn(*mut c_char, usize) -> i32,
{
    let mut buffer = vec![0u8; capacity];

    // call the foreign function
    let result = c_function(buffer.as_mut_ptr().cast::<c_char>(), buffer.len());
    if result < 0 {
        return Err(MetaValError::NegativeReturn(result));
    }

    // check if the response fit in our buffer
    let returned_len = result as usize;
    if returned_len >= capacity {
        // buffer wasn't large enough, try again with the correct capacity.
        return extract_meta_string(c_function, returned_len + 1);
    }

    // verify null termination
    debug_assert_eq!(
        buffer.get(returned_len),
        Some(&0),
        "should end with null byte"
    );

    // resize, convert, and return
    buffer.truncate(returned_len);
    Ok(String::from_utf8(buffer)?)
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_free_model(self.model.as_mut_ptr()) }
    }
}
