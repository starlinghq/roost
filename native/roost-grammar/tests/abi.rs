//! Tests of the C ABI through the rlib, on the tiny byte-level tokenizer the .NET tests use (264 ids, 263 is `<|im_end|>`).

use std::ptr;

use roost_grammar::*;

const TOKENIZER: &[u8] =
    include_bytes!("../../../tests/Starlings.Roost.Generation.Tests/Fixtures/tiny-bytelevel-bpe.tokenizer.json");
const STOP: u32 = 263;

struct Tok(*mut RoostGrammarTokenizer);

impl Drop for Tok {
    fn drop(&mut self) {
        unsafe { roost_grammar_tokenizer_free(self.0) }
    }
}

struct M(*mut RoostGrammarMatcher);

impl Drop for M {
    fn drop(&mut self) {
        unsafe { roost_grammar_matcher_free(self.0) }
    }
}

fn last_error() -> String {
    let mut buf = vec![0u8; 1024];
    let n = unsafe { roost_grammar_last_error(buf.as_mut_ptr().cast(), buf.len()) };
    String::from_utf8_lossy(&buf[..n.min(buf.len() - 1)]).into_owned()
}

fn tokenizer(n_vocab: u32) -> Tok {
    let mut t = ptr::null_mut();
    let stops = [STOP];
    let rc = unsafe { roost_grammar_tokenizer_new(TOKENIZER.as_ptr(), TOKENIZER.len(), n_vocab, stops.as_ptr(), 1, &mut t) };
    assert_eq!(rc, ROOST_GRAMMAR_OK, "{}", last_error());
    Tok(t)
}

fn matcher(t: &Tok, kind: i32, data: &str) -> Result<M, (i32, String)> {
    let mut m = ptr::null_mut();
    let rc = unsafe { roost_grammar_matcher_new(t.0, kind, data.as_ptr(), data.len(), &mut m) };
    if rc == ROOST_GRAMMAR_OK {
        Ok(M(m))
    } else {
        assert!(m.is_null());
        Err((rc, last_error()))
    }
}

fn tokenize(t: &Tok, text: &str) -> Vec<u32> {
    let mut count = 0usize;
    let mut out = vec![0u32; text.len() + 8];
    let rc = unsafe { roost_grammar_tokenize(t.0, text.as_ptr(), text.len(), out.as_mut_ptr(), out.len(), &mut count) };
    assert_eq!(rc, ROOST_GRAMMAR_OK);
    out.truncate(count);
    out
}

fn mask(t: &Tok, m: &M) -> Vec<u32> {
    let words = unsafe { roost_grammar_mask_words(t.0) };
    let mut dest = vec![0u32; words];
    let rc = unsafe { roost_grammar_matcher_compute_mask(m.0, dest.as_mut_ptr(), words) };
    assert_eq!(rc, ROOST_GRAMMAR_OK, "{}", last_error());
    dest
}

fn allowed(mask: &[u32], token: u32) -> bool {
    (mask[(token / 32) as usize] >> (token % 32)) & 1 == 1
}

/// Replays `text` through the matcher, checking every token is allowed by the mask before it is consumed.
fn replay(t: &Tok, m: &M, text: &str) {
    for tok in tokenize(t, text) {
        let mk = mask(t, m);
        assert!(allowed(&mk, tok), "token {tok} of {text:?} masked");
        assert_eq!(unsafe { roost_grammar_matcher_consume(m.0, tok) }, ROOST_GRAMMAR_OK, "{}", last_error());
    }
}

#[test]
fn abi_version_is_one_zero() {
    assert_eq!(roost_grammar_abi_version(), 0x0001_0000);
}

#[test]
fn tokenizer_reports_vocab_and_padding() {
    let t = tokenizer(0);
    assert_eq!(unsafe { roost_grammar_tokenizer_vocab(t.0) }, 264);
    assert_eq!(unsafe { roost_grammar_mask_words(t.0) }, 9);
    let padded = tokenizer(300);
    assert_eq!(unsafe { roost_grammar_tokenizer_vocab(padded.0) }, 300);
    assert_eq!(unsafe { roost_grammar_mask_words(padded.0) }, 10);
}

#[test]
fn tokenizer_refuses_small_vocab_and_bad_stop() {
    let mut t = ptr::null_mut();
    let rc = unsafe { roost_grammar_tokenizer_new(TOKENIZER.as_ptr(), TOKENIZER.len(), 100, ptr::null(), 0, &mut t) };
    assert_eq!(rc, ROOST_GRAMMAR_E_ARGUMENT);
    assert!(t.is_null());
    let stops = [5000u32];
    let rc = unsafe { roost_grammar_tokenizer_new(TOKENIZER.as_ptr(), TOKENIZER.len(), 0, stops.as_ptr(), 1, &mut t) };
    assert_eq!(rc, ROOST_GRAMMAR_E_ARGUMENT);
    let junk = b"not json";
    let rc = unsafe { roost_grammar_tokenizer_new(junk.as_ptr(), junk.len(), 0, ptr::null(), 0, &mut t) };
    assert_eq!(rc, ROOST_GRAMMAR_E_TOKENIZER);
    assert!(!last_error().is_empty());
}

#[test]
fn json_schema_accepts_instance_then_only_stop() {
    let t = tokenizer(0);
    let schema = r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"enum":["x","y"]}},"required":["a","b"],"additionalProperties":false}"#;
    let m = matcher(&t, ROOST_GRAMMAR_KIND_JSON_SCHEMA, schema).unwrap();
    assert_eq!(unsafe { roost_grammar_matcher_is_accepting(m.0) }, 0);
    replay(&t, &m, r#"{"a":12,"b":"y"}"#);
    assert_eq!(unsafe { roost_grammar_matcher_is_accepting(m.0) }, 1);
    assert_eq!(unsafe { roost_grammar_matcher_is_stopped(m.0) }, 1);
    let mk = mask(&t, &m);
    let set: Vec<u32> = (0..264).filter(|&v| allowed(&mk, v)).collect();
    assert_eq!(set, vec![STOP]);
    // The stop id is accepted without being consumed; anything else is refused.
    assert_eq!(unsafe { roost_grammar_matcher_consume(m.0, STOP) }, ROOST_GRAMMAR_OK);
    assert_eq!(unsafe { roost_grammar_matcher_consume(m.0, u32::from(b'x')) }, ROOST_GRAMMAR_E_REJECTED);
}

#[test]
fn json_is_compact_by_default() {
    let t = tokenizer(0);
    let m = matcher(&t, ROOST_GRAMMAR_KIND_JSON_OBJECT, "").unwrap();
    replay(&t, &m, "{");
    let mk = mask(&t, &m);
    assert!(allowed(&mk, u32::from(b'"')));
    assert!(allowed(&mk, u32::from(b'}')));
    assert!(!allowed(&mk, 32), "a space is masked"); // ids are byte values below 256
    assert!(!allowed(&mk, STOP));
}

#[test]
fn rejected_token_leaves_matcher_usable() {
    let t = tokenizer(0);
    let m = matcher(&t, ROOST_GRAMMAR_KIND_JSON_OBJECT, "").unwrap();
    let first = mask(&t, &m);
    assert!(!allowed(&first, u32::from(b'a')));
    assert_eq!(unsafe { roost_grammar_matcher_consume(m.0, u32::from(b'a')) }, ROOST_GRAMMAR_E_REJECTED);
    assert_eq!(unsafe { roost_grammar_matcher_consume(m.0, 999) }, ROOST_GRAMMAR_E_ARGUMENT);
    assert_eq!(mask(&t, &m), first);
    replay(&t, &m, "{}");
    assert_eq!(unsafe { roost_grammar_matcher_is_accepting(m.0) }, 1);
}

#[test]
fn regex_and_lark_kinds() {
    let t = tokenizer(0);
    let r = matcher(&t, ROOST_GRAMMAR_KIND_REGEX, "(yes|no)[0-9]{2}").unwrap();
    replay(&t, &r, "no42");
    assert_eq!(unsafe { roost_grammar_matcher_is_accepting(r.0) }, 1);
    let l = matcher(&t, ROOST_GRAMMAR_KIND_LARK, "start: \"a\" \"b\"+").unwrap();
    replay(&t, &l, "abb");
    assert_eq!(unsafe { roost_grammar_matcher_is_accepting(l.0) }, 1);
}

#[test]
fn invalid_grammars_are_grammar_errors() {
    let t = tokenizer(0);
    let (rc, msg) = matcher(&t, ROOST_GRAMMAR_KIND_JSON_SCHEMA, r#"{"type":"array","uniqueItems":true}"#).err().unwrap();
    assert_eq!(rc, ROOST_GRAMMAR_E_GRAMMAR);
    assert!(msg.contains("uniqueItems"), "{msg}");
    let (rc, _) = matcher(&t, ROOST_GRAMMAR_KIND_JSON_SCHEMA, "{not json").err().unwrap();
    assert_eq!(rc, ROOST_GRAMMAR_E_GRAMMAR);
    let (rc, _) = matcher(&t, ROOST_GRAMMAR_KIND_JSON_SCHEMA, "[1]").err().unwrap();
    assert_eq!(rc, ROOST_GRAMMAR_E_GRAMMAR);
    let (rc, _) = matcher(&t, ROOST_GRAMMAR_KIND_LARK, "start: undefined_rule").err().unwrap();
    assert_eq!(rc, ROOST_GRAMMAR_E_GRAMMAR);
    let (rc, _) = matcher(&t, 99, "").err().unwrap();
    assert_eq!(rc, ROOST_GRAMMAR_E_ARGUMENT);
}

#[test]
fn mask_size_and_null_arguments_are_checked() {
    let t = tokenizer(0);
    let m = matcher(&t, ROOST_GRAMMAR_KIND_JSON_OBJECT, "").unwrap();
    let mut small = [0u32; 3];
    assert_eq!(unsafe { roost_grammar_matcher_compute_mask(m.0, small.as_mut_ptr(), 3) }, ROOST_GRAMMAR_E_ARGUMENT);
    assert_eq!(unsafe { roost_grammar_matcher_consume(ptr::null_mut(), 1) }, ROOST_GRAMMAR_E_ARGUMENT);
    assert_eq!(unsafe { roost_grammar_matcher_is_stopped(ptr::null()) }, ROOST_GRAMMAR_E_ARGUMENT);
    assert_eq!(unsafe { roost_grammar_tokenizer_vocab(ptr::null()) }, 0);
}

#[test]
fn padded_rows_are_never_allowed() {
    let t = tokenizer(300);
    let m = matcher(&t, ROOST_GRAMMAR_KIND_REGEX, "[a-z]+").unwrap();
    let mk = mask(&t, &m);
    assert!((264..300).all(|v| !allowed(&mk, v)));
    assert_eq!(mk[9] >> 12, 0, "bits past the vocabulary are clear");
}

#[test]
fn matcher_outlives_its_tokenizer() {
    let t = tokenizer(0);
    let m = matcher(&t, ROOST_GRAMMAR_KIND_REGEX, "ab").unwrap();
    let words = unsafe { roost_grammar_mask_words(t.0) };
    drop(t);
    let mut dest = vec![0u32; words];
    assert_eq!(unsafe { roost_grammar_matcher_compute_mask(m.0, dest.as_mut_ptr(), words) }, ROOST_GRAMMAR_OK);
    assert!(allowed(&dest, u32::from(b'a')));
}
