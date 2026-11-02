//! roost-grammar: llguidance token masks behind Roost's versioned C ABI (R-051, decision 22).
//!
//! Promoted from spike S10 (`spikes/s10-llguidance`, `docs/m0/roost-s10-llguidance-abi.md`). The surface is Roost's own
//! (`include/roost_grammar.h`), not llguidance's `llg_*` C API, so Roost versions it. Handles are opaque; every fallible
//! function returns a `ROOST_GRAMMAR_*` status and `roost_grammar_last_error` copies the calling thread's last message.
//! Panics never cross the boundary.
//!
//! Behaviour taken from the S10 findings:
//! - the model's stop ids are given to the tokenizer (`set_eos_tokens`), because a tokenizer's own EOS is not always the
//!   model's generation stop id (Qwen3: 151643 against 151645);
//! - a matcher stops itself after a complete value, and the stop token that follows is then accepted without being consumed;
//! - an unsupported schema keyword or an invalid grammar is `ROOST_GRAMMAR_E_GRAMMAR`, which callers report as the
//!   request's fault (HTTP 400), not an internal error.

use std::cell::RefCell;
use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{TokEnv, TokenId};
use llguidance::{Matcher, ParserFactory};
use serde_json::Value;
use toktrie_hf_tokenizers::ByteTokenizer;

/// C ABI version, `(major << 16) | minor`. A minor bump adds functions; a major bump changes or removes one.
pub const ABI_VERSION: u32 = 1 << 16;

/// Success.
pub const ROOST_GRAMMAR_OK: i32 = 0;
/// A null pointer, a size that does not match, an unknown kind, a token id out of range or text that is not UTF-8.
pub const ROOST_GRAMMAR_E_ARGUMENT: i32 = -1;
/// The grammar or schema is invalid or uses something the matcher does not implement (the caller's input).
pub const ROOST_GRAMMAR_E_GRAMMAR: i32 = -2;
/// The tokenizer description could not be read.
pub const ROOST_GRAMMAR_E_TOKENIZER: i32 = -3;
/// The token is not allowed in the matcher's state; the matcher is unchanged.
pub const ROOST_GRAMMAR_E_REJECTED: i32 = -4;
/// The matcher is in its error state and can only be freed.
pub const ROOST_GRAMMAR_E_STATE: i32 = -5;
/// A panic was caught at the boundary.
pub const ROOST_GRAMMAR_E_PANIC: i32 = -6;

/// A JSON schema (draft 2020-12 subset that llguidance implements).
pub const ROOST_GRAMMAR_KIND_JSON_SCHEMA: i32 = 1;
/// Any JSON object (`{"type":"object"}`); `data` is ignored.
pub const ROOST_GRAMMAR_KIND_JSON_OBJECT: i32 = 2;
/// A regular expression over the whole output.
pub const ROOST_GRAMMAR_KIND_REGEX: i32 = 3;
/// A Lark grammar (llguidance's dialect).
pub const ROOST_GRAMMAR_KIND_LARK: i32 = 4;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_err(e: impl std::fmt::Display) {
    LAST_ERROR.with(|l| *l.borrow_mut() = e.to_string());
}

/// A failure with its status code.
struct Fail(i32, String);

impl Fail {
    fn arg(msg: impl Into<String>) -> Self {
        Fail(ROOST_GRAMMAR_E_ARGUMENT, msg.into())
    }
}

type Res<T> = Result<T, Fail>;

/// Runs `f` with panics caught; stores the message of a failure and returns its status.
fn guard(f: impl FnOnce() -> Res<i32>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(Fail(code, msg))) => {
            set_err(msg);
            code
        }
        Err(_) => {
            set_err("panic in roost-grammar");
            ROOST_GRAMMAR_E_PANIC
        }
    }
}

/// Tokenizer environment plus the parser factory (slicer tables) built from it; shared by every matcher of one model.
pub struct RoostGrammarTokenizer {
    env: TokEnv,
    factory: ParserFactory,
    vocab: usize,
    mask_words: usize,
    stop_ids: Vec<TokenId>,
}

/// One grammar's matcher for one sequence.
pub struct RoostGrammarMatcher {
    matcher: Matcher,
    vocab: usize,
    mask_words: usize,
    stop_ids: Vec<TokenId>,
}

/// The C ABI version, `(major << 16) | minor`.
#[no_mangle]
pub extern "C" fn roost_grammar_abi_version() -> u32 {
    ABI_VERSION
}

/// Copies the calling thread's last error message into `buf` (NUL-terminated, truncated to `cap - 1` bytes) and returns
/// its full length in bytes. `buf` may be null to ask for the length.
///
/// # Safety
/// `buf` must be valid for `cap` bytes when it is not null.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_last_error(buf: *mut c_char, cap: usize) -> usize {
    LAST_ERROR.with(|l| {
        let s = l.borrow();
        if !buf.is_null() && cap > 0 {
            let n = s.len().min(cap - 1);
            std::ptr::copy_nonoverlapping(s.as_ptr(), buf.cast::<u8>(), n);
            *buf.add(n) = 0;
        }
        s.len()
    })
}

/// Builds a tokenizer environment from `tokenizer.json` bytes.
///
/// `n_vocab` pads the vocabulary to the model's logits width (0 keeps the tokenizer's own size; a value below it is an
/// argument error). `stop_ids` (`stop_count` entries, may be 0) are the model's generation stop ids; when given they
/// replace the tokenizer's EOS, and a matcher allows them once its grammar is complete.
///
/// # Safety
/// `json` must be valid for `len` bytes, `stop_ids` for `stop_count` values, and `out` writable.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_tokenizer_new(
    json: *const u8,
    len: usize,
    n_vocab: u32,
    stop_ids: *const u32,
    stop_count: usize,
    out: *mut *mut RoostGrammarTokenizer,
) -> i32 {
    guard(|| {
        if out.is_null() || json.is_null() || (stop_count > 0 && stop_ids.is_null()) {
            return Err(Fail::arg("null argument"));
        }
        *out = std::ptr::null_mut();
        let bytes = std::slice::from_raw_parts(json, len);
        let mut tok = ByteTokenizer::from_json_bytes(bytes)
            .map_err(|e| Fail(ROOST_GRAMMAR_E_TOKENIZER, e.to_string()))?;
        let own = tok.tokrx_info().vocab_size as usize;
        let nv = if n_vocab == 0 { own } else { n_vocab as usize };
        if nv < own {
            return Err(Fail::arg(format!(
                "n_vocab {nv} is smaller than the tokenizer's {own} tokens"
            )));
        }
        let stops: Vec<TokenId> = if stop_count == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(stop_ids, stop_count).to_vec()
        };
        if let Some(bad) = stops.iter().find(|&&s| s as usize >= own) {
            return Err(Fail::arg(format!(
                "stop id {bad} is outside the tokenizer's {own} tokens"
            )));
        }
        if !stops.is_empty() {
            tok.set_eos_tokens(&stops);
        }
        let stop_ids = tok.eos_tokens();
        let env = tok
            .into_tok_env(Some(nv))
            .map_err(|e| Fail(ROOST_GRAMMAR_E_TOKENIZER, e.to_string()))?;
        let factory =
            ParserFactory::new_simple(&env).map_err(|e| Fail(ROOST_GRAMMAR_E_TOKENIZER, e.to_string()))?;
        let vocab = env.tok_trie().vocab_size();
        *out = Box::into_raw(Box::new(RoostGrammarTokenizer {
            env,
            factory,
            vocab,
            mask_words: vocab.div_ceil(32),
            stop_ids,
        }));
        Ok(ROOST_GRAMMAR_OK)
    })
}

/// Frees a tokenizer. Matchers made from it keep what they need and stay valid.
///
/// # Safety
/// `t` must come from `roost_grammar_tokenizer_new` (or be null) and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_tokenizer_free(t: *mut RoostGrammarTokenizer) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// The vocabulary size (mask rows) of a tokenizer, after padding; 0 for a null handle.
///
/// # Safety
/// `t` must be a live tokenizer handle or null.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_tokenizer_vocab(t: *const RoostGrammarTokenizer) -> u32 {
    if t.is_null() {
        0
    } else {
        (*t).vocab as u32
    }
}

/// Number of u32 words in a mask for this tokenizer, `ceil(vocab / 32)`; 0 for a null handle.
///
/// # Safety
/// `t` must be a live tokenizer handle or null.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_mask_words(t: *const RoostGrammarTokenizer) -> usize {
    if t.is_null() {
        0
    } else {
        (*t).mask_words
    }
}

/// Tokenizes UTF-8 text with llguidance's greedy byte tokenization (for tests and tools; it need not match the model's
/// own tokenizer). Writes up to `cap` ids to `out` (may be null) and stores the full count in `count`.
///
/// # Safety
/// `text` valid for `len` bytes, `out` for `cap` values when not null, `count` writable.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_tokenize(
    t: *const RoostGrammarTokenizer,
    text: *const u8,
    len: usize,
    out: *mut u32,
    cap: usize,
    count: *mut usize,
) -> i32 {
    guard(|| {
        if t.is_null() || text.is_null() || count.is_null() {
            return Err(Fail::arg("null argument"));
        }
        let toks = (*t).env.tokenize_bytes(std::slice::from_raw_parts(text, len));
        if !out.is_null() {
            let n = toks.len().min(cap);
            std::ptr::copy_nonoverlapping(toks.as_ptr(), out, n);
        }
        *count = toks.len();
        Ok(ROOST_GRAMMAR_OK)
    })
}

/// The JSON schema for a JSON kind, with compact separators unless the schema sets its own `x-guidance` options: llguidance
/// otherwise allows free whitespace between tokens, which lets a model emit whitespace until it runs out of tokens.
fn json_grammar(kind: i32, data: &str) -> Res<TopLevelGrammar> {
    let mut schema: Value = if kind == ROOST_GRAMMAR_KIND_JSON_OBJECT {
        serde_json::json!({ "type": "object" })
    } else {
        serde_json::from_str(data)
            .map_err(|e| Fail(ROOST_GRAMMAR_E_GRAMMAR, format!("the schema is not JSON: {e}")))?
    };
    match &mut schema {
        Value::Object(map) => {
            map.entry("x-guidance")
                .or_insert_with(|| serde_json::json!({ "whitespace_flexible": false }));
        }
        Value::Bool(_) => {}
        _ => {
            return Err(Fail(
                ROOST_GRAMMAR_E_GRAMMAR,
                "the schema must be a JSON object or a boolean".into(),
            ))
        }
    }
    Ok(TopLevelGrammar::from_json_schema(schema))
}

/// Compiles a grammar into a matcher for one sequence.
///
/// `kind` is a `ROOST_GRAMMAR_KIND_*`; `data` (`len` bytes of UTF-8, not NUL-terminated) is the schema, regex or Lark
/// source, and is ignored for `ROOST_GRAMMAR_KIND_JSON_OBJECT`. An invalid or unsupported grammar is
/// `ROOST_GRAMMAR_E_GRAMMAR` with the reason in `roost_grammar_last_error` (for example `Unimplemented keys: ["uniqueItems"]`).
///
/// # Safety
/// `t` must be live, `data` valid for `len` bytes (may be null when `len` is 0), `out` writable.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_matcher_new(
    t: *const RoostGrammarTokenizer,
    kind: i32,
    data: *const u8,
    len: usize,
    out: *mut *mut RoostGrammarMatcher,
) -> i32 {
    guard(|| {
        if t.is_null() || out.is_null() || (data.is_null() && len > 0) {
            return Err(Fail::arg("null argument"));
        }
        *out = std::ptr::null_mut();
        let t = &*t;
        let text = if len == 0 {
            ""
        } else {
            std::str::from_utf8(std::slice::from_raw_parts(data, len))
                .map_err(|_| Fail::arg("the grammar is not UTF-8"))?
        };
        let grammar = match kind {
            ROOST_GRAMMAR_KIND_JSON_SCHEMA | ROOST_GRAMMAR_KIND_JSON_OBJECT => json_grammar(kind, text)?,
            ROOST_GRAMMAR_KIND_REGEX => TopLevelGrammar::from_regex(text),
            ROOST_GRAMMAR_KIND_LARK => TopLevelGrammar::from_lark(text.to_string()),
            _ => return Err(Fail::arg(format!("unknown grammar kind {kind}"))),
        };
        let parser = t
            .factory
            .create_parser(grammar)
            .map_err(|e| Fail(ROOST_GRAMMAR_E_GRAMMAR, e.to_string()))?;
        let matcher = Matcher::new(Ok(parser));
        if let Some(e) = matcher.get_error() {
            return Err(Fail(ROOST_GRAMMAR_E_GRAMMAR, e));
        }
        *out = Box::into_raw(Box::new(RoostGrammarMatcher {
            matcher,
            vocab: t.vocab,
            mask_words: t.mask_words,
            stop_ids: t.stop_ids.clone(),
        }));
        Ok(ROOST_GRAMMAR_OK)
    })
}

/// Frees a matcher.
///
/// # Safety
/// `m` must come from `roost_grammar_matcher_new` (or be null) and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_matcher_free(m: *mut RoostGrammarMatcher) {
    if !m.is_null() {
        drop(Box::from_raw(m));
    }
}

fn state_error(m: &RoostGrammarMatcher) -> Fail {
    Fail(
        ROOST_GRAMMAR_E_STATE,
        m.matcher
            .get_error()
            .unwrap_or_else(|| "the matcher is in its error state".into()),
    )
}

/// Writes the allowed-token bitmask for the next token into `dest`: bit `v % 32` of word `v / 32` is 1 when token `v`
/// may follow. Once the grammar is complete and the matcher has stopped, only the stop ids are allowed. `words` must equal
/// `roost_grammar_mask_words`.
///
/// # Safety
/// `m` must be live and `dest` valid for `words` values.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_matcher_compute_mask(
    m: *mut RoostGrammarMatcher,
    dest: *mut u32,
    words: usize,
) -> i32 {
    guard(|| {
        if m.is_null() || dest.is_null() {
            return Err(Fail::arg("null argument"));
        }
        let m = &mut *m;
        if words != m.mask_words {
            return Err(Fail::arg(format!(
                "mask has {words} words; the tokenizer needs {}",
                m.mask_words
            )));
        }
        if m.matcher.is_error() {
            return Err(state_error(m));
        }
        let vob = m.matcher.compute_mask_or_eos().map_err(|e| Fail(ROOST_GRAMMAR_E_STATE, e.to_string()))?;
        let src = vob.as_slice();
        let n = src.len().min(words);
        std::ptr::copy_nonoverlapping(src.as_ptr(), dest, n);
        if n < words {
            std::ptr::write_bytes(dest.add(n), 0, words - n);
        }
        // Clear the bits past the vocabulary in the last word, so a caller may test a whole word.
        let tail = m.vocab % 32;
        if tail != 0 {
            *dest.add(words - 1) &= (1u32 << tail) - 1;
        }
        Ok(ROOST_GRAMMAR_OK)
    })
}

/// Advances the matcher by one sampled token. A token the grammar does not allow is `ROOST_GRAMMAR_E_REJECTED` and leaves
/// the matcher as it was. A stop id after the matcher has stopped is accepted and changes nothing (S10 finding 2).
///
/// # Safety
/// `m` must be live.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_matcher_consume(m: *mut RoostGrammarMatcher, token: u32) -> i32 {
    guard(|| {
        if m.is_null() {
            return Err(Fail::arg("null argument"));
        }
        let m = &mut *m;
        if token as usize >= m.vocab {
            return Err(Fail::arg(format!(
                "token {token} is outside the vocabulary of {}",
                m.vocab
            )));
        }
        if m.matcher.is_error() {
            return Err(state_error(m));
        }
        if m.matcher.is_stopped() {
            return if m.stop_ids.contains(&token) {
                Ok(ROOST_GRAMMAR_OK)
            } else {
                Err(Fail(
                    ROOST_GRAMMAR_E_REJECTED,
                    format!("token {token} after the grammar completed; only a stop id may follow"),
                ))
            };
        }
        let valid = m
            .matcher
            .validate_tokens(&[token])
            .map_err(|e| Fail(ROOST_GRAMMAR_E_STATE, e.to_string()))?;
        if valid != 1 {
            return Err(Fail(
                ROOST_GRAMMAR_E_REJECTED,
                format!("token {token} is not allowed by the grammar here"),
            ));
        }
        m.matcher
            .consume_token(token)
            .map_err(|e| Fail(ROOST_GRAMMAR_E_STATE, e.to_string()))?;
        Ok(ROOST_GRAMMAR_OK)
    })
}

/// 1 when the output so far is a complete sentence of the grammar (a stop id is allowed next), 0 when not, or a negative
/// status.
///
/// # Safety
/// `m` must be live.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_matcher_is_accepting(m: *mut RoostGrammarMatcher) -> i32 {
    guard(|| {
        if m.is_null() {
            return Err(Fail::arg("null argument"));
        }
        let m = &mut *m;
        if m.matcher.is_error() {
            return Err(state_error(m));
        }
        if m.matcher.is_stopped() {
            return Ok(1);
        }
        m.matcher
            .is_accepting()
            .map(i32::from)
            .map_err(|e| Fail(ROOST_GRAMMAR_E_STATE, e.to_string()))
    })
}

/// 1 when the matcher has stopped (the grammar is complete and nothing but a stop id may follow), 0 when not, or a negative
/// status (the error state).
///
/// # Safety
/// `m` must be live.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_matcher_is_stopped(m: *const RoostGrammarMatcher) -> i32 {
    guard(|| {
        if m.is_null() {
            return Err(Fail::arg("null argument"));
        }
        let m = &*m;
        if m.matcher.is_error() {
            return Err(state_error(m));
        }
        Ok(i32::from(m.matcher.is_stopped()))
    })
}
