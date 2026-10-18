//! Spike S10: a minimal C ABI over llguidance token masks.
//!
//! Handles are opaque. Every function returns a negative value or null on
//! failure; `roost_grammar_last_error` copies the thread's last message.
//! Panics never cross the boundary.

use std::cell::RefCell;
use std::ffi::{c_char, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::TokEnv;
use llguidance::{Matcher, ParserFactory};
use toktrie_hf_tokenizers::ByteTokenizer;

pub const ABI_VERSION: u32 = 1;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_err(e: impl std::fmt::Display) {
    LAST_ERROR.with(|l| *l.borrow_mut() = e.to_string());
}

/// Tokenizer environment plus the parser factory (slicer tables) built from it.
pub struct RgTokenizer {
    env: TokEnv,
    factory: ParserFactory,
    mask_words: usize,
}

pub struct RgMatcher {
    matcher: Matcher,
    mask_words: usize,
}

fn guard<T>(fail: T, f: impl FnOnce() -> anyhow::Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_err(e);
            fail
        }
        Err(_) => {
            set_err("panic in roost-grammar");
            fail
        }
    }
}

#[no_mangle]
pub extern "C" fn roost_grammar_abi_version() -> u32 {
    ABI_VERSION
}

/// Copy the last error into `buf` (NUL-terminated, truncated). Returns the full length.
///
/// # Safety
/// `buf` must be valid for `cap` bytes.
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

/// Build a tokenizer environment from `tokenizer.json` bytes. `n_vocab` pads the
/// vocabulary to the model's embedding rows (0 keeps the tokenizer's own size).
///
/// # Safety
/// `json` must be valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_tokenizer_new(
    json: *const u8,
    len: usize,
    n_vocab: u32,
) -> *mut RgTokenizer {
    guard(std::ptr::null_mut(), || {
        let bytes = std::slice::from_raw_parts(json, len);
        let tok = ByteTokenizer::from_json_bytes(bytes)?;
        let nv = (n_vocab != 0).then_some(n_vocab as usize);
        let env = tok.into_tok_env(nv)?;
        let factory = ParserFactory::new_simple(&env)?;
        let mask_words = env.tok_trie().vocab_size().div_ceil(32);
        Ok(Box::into_raw(Box::new(RgTokenizer {
            env,
            factory,
            mask_words,
        })))
    })
}

/// # Safety
/// `t` must come from `roost_grammar_tokenizer_new` and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_tokenizer_free(t: *mut RgTokenizer) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// Number of u32 words in a mask for this tokenizer.
///
/// # Safety
/// `t` must be a live tokenizer handle.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_mask_words(t: *const RgTokenizer) -> usize {
    (*t).mask_words
}

/// Tokenize UTF-8 text; returns the token count (may exceed `cap`), or -1.
///
/// # Safety
/// `text` valid for `len` bytes, `out` for `cap` u32s.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_tokenize(
    t: *const RgTokenizer,
    text: *const u8,
    len: usize,
    out: *mut u32,
    cap: usize,
) -> isize {
    guard(-1, || {
        let toks = (*t).env.tokenize_bytes(std::slice::from_raw_parts(text, len));
        if !out.is_null() {
            let n = toks.len().min(cap);
            std::ptr::copy_nonoverlapping(toks.as_ptr(), out, n);
        }
        Ok(toks.len() as isize)
    })
}

/// EOS token id of the tokenizer.
///
/// # Safety
/// `t` must be a live tokenizer handle.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_eos_token(t: *const RgTokenizer) -> u32 {
    (*t).env.tok_trie().eos_token()
}

/// Create a matcher. `kind` is "json_schema", "regex", "lark" or "json_object".
/// Returns null on failure (see `roost_grammar_last_error`).
///
/// # Safety
/// `kind` and `data` are NUL-terminated UTF-8; `t` is live.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_matcher_new(
    t: *const RgTokenizer,
    kind: *const c_char,
    data: *const c_char,
) -> *mut RgMatcher {
    guard(std::ptr::null_mut(), || {
        let t = &*t;
        let kind = CStr::from_ptr(kind).to_str()?;
        let data = CStr::from_ptr(data).to_str()?;
        let grammar = TopLevelGrammar::from_tagged_str(kind, data)?;
        let parser = t.factory.create_parser(grammar)?;
        let matcher = Matcher::new(Ok(parser));
        if matcher.is_error() {
            anyhow::bail!("matcher construction failed");
        }
        Ok(Box::into_raw(Box::new(RgMatcher {
            matcher,
            mask_words: t.mask_words,
        })))
    })
}

/// # Safety
/// `m` must come from `roost_grammar_matcher_new` and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_matcher_free(m: *mut RgMatcher) {
    if !m.is_null() {
        drop(Box::from_raw(m));
    }
}

/// Write the allowed-token bitmask (bit i of word i/32) into `dest`.
/// `words` must equal `roost_grammar_mask_words`. Returns 0 or -1.
///
/// # Safety
/// `dest` valid for `words` u32s.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_matcher_compute_mask(
    m: *mut RgMatcher,
    dest: *mut u32,
    words: usize,
) -> i32 {
    guard(-1, || {
        let m = &mut *m;
        anyhow::ensure!(words == m.mask_words, "mask size mismatch");
        let vob = m.matcher.compute_mask_or_eos()?;
        let src = vob.as_slice();
        std::ptr::copy_nonoverlapping(src.as_ptr(), dest, words);
        Ok(0)
    })
}

/// Advance the matcher by one sampled token. Returns 0 or -1.
///
/// # Safety
/// `m` must be live.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_matcher_consume(m: *mut RgMatcher, token: u32) -> i32 {
    guard(-1, || {
        (*m).matcher.consume_token(token)?;
        Ok(0)
    })
}

/// 1 when the grammar is complete after the tokens consumed so far, 0 if not, -1 on error.
///
/// # Safety
/// `m` must be live.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_matcher_is_accepting(m: *mut RgMatcher) -> i32 {
    guard(-1, || Ok(i32::from((*m).matcher.is_accepting()?)))
}

/// 1 once the matcher has consumed EOS and stopped.
///
/// # Safety
/// `m` must be live.
#[no_mangle]
pub unsafe extern "C" fn roost_grammar_matcher_is_stopped(m: *const RgMatcher) -> i32 {
    i32::from((*m).matcher.is_stopped())
}
