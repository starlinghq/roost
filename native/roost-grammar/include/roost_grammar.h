/*
 * roost_grammar.h: the C ABI of native/roost-grammar (R-051, decision 22), llguidance token masks for structured output.
 *
 * Version: ROOST_GRAMMAR_ABI_VERSION, (major << 16) | minor. A minor bump only adds functions; a major bump changes or
 * removes one. Loaders refuse a library whose major differs or whose minor is older than they need.
 *
 * Rules:
 *   - Handles are opaque. A tokenizer is built once per model and shared; a matcher is one grammar for one sequence.
 *     A matcher keeps what it needs from its tokenizer, so freeing the tokenizer first is allowed.
 *   - A matcher is not thread safe; different matchers may be used on different threads at the same time.
 *   - Fallible functions return a ROOST_GRAMMAR_* status (0 is success). roost_grammar_last_error copies the calling
 *     thread's last message. Panics are caught at the boundary and reported as ROOST_GRAMMAR_E_PANIC.
 *   - Masks are uint32 words: bit (v % 32) of word (v / 32) is 1 when token v is allowed; bits past the vocabulary are 0.
 *   - JSON kinds use compact separators (no free whitespace) unless the schema carries its own "x-guidance" options.
 */
#ifndef ROOST_GRAMMAR_H_
#define ROOST_GRAMMAR_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROOST_GRAMMAR_ABI_VERSION 0x00010000u

#define ROOST_GRAMMAR_OK 0
#define ROOST_GRAMMAR_E_ARGUMENT (-1)  /* null pointer, size mismatch, unknown kind, token out of range, not UTF-8 */
#define ROOST_GRAMMAR_E_GRAMMAR (-2)   /* invalid or unsupported grammar or schema: the caller's input */
#define ROOST_GRAMMAR_E_TOKENIZER (-3) /* the tokenizer description could not be read */
#define ROOST_GRAMMAR_E_REJECTED (-4)  /* the token is not allowed here; the matcher is unchanged */
#define ROOST_GRAMMAR_E_STATE (-5)     /* the matcher is in its error state; free it */
#define ROOST_GRAMMAR_E_PANIC (-6)     /* a panic was caught at the boundary */

#define ROOST_GRAMMAR_KIND_JSON_SCHEMA 1 /* data: a JSON schema */
#define ROOST_GRAMMAR_KIND_JSON_OBJECT 2 /* any JSON object; data ignored */
#define ROOST_GRAMMAR_KIND_REGEX 3       /* data: a regular expression over the whole output */
#define ROOST_GRAMMAR_KIND_LARK 4        /* data: a Lark grammar (llguidance dialect) */

typedef struct RoostGrammarTokenizer RoostGrammarTokenizer;
typedef struct RoostGrammarMatcher RoostGrammarMatcher;

uint32_t roost_grammar_abi_version(void);

/* Copies the last error of the calling thread into buf (NUL-terminated, truncated); returns its full length. */
size_t roost_grammar_last_error(char* buf, size_t cap);

/* tokenizer.json bytes; n_vocab pads to the model's logits width (0 keeps the tokenizer's size); stop_ids replace the
 * tokenizer's EOS when stop_count > 0. */
int32_t roost_grammar_tokenizer_new(const uint8_t* json, size_t len, uint32_t n_vocab, const uint32_t* stop_ids,
                                    size_t stop_count, RoostGrammarTokenizer** out);
void roost_grammar_tokenizer_free(RoostGrammarTokenizer* tokenizer);
uint32_t roost_grammar_tokenizer_vocab(const RoostGrammarTokenizer* tokenizer);
size_t roost_grammar_mask_words(const RoostGrammarTokenizer* tokenizer);

/* llguidance's greedy byte tokenization, for tests and tools. Writes up to cap ids; *count receives the full count. */
int32_t roost_grammar_tokenize(const RoostGrammarTokenizer* tokenizer, const uint8_t* text, size_t len, uint32_t* out,
                               size_t cap, size_t* count);

/* data is len bytes of UTF-8, not NUL-terminated. */
int32_t roost_grammar_matcher_new(const RoostGrammarTokenizer* tokenizer, int32_t kind, const uint8_t* data, size_t len,
                                  RoostGrammarMatcher** out);
void roost_grammar_matcher_free(RoostGrammarMatcher* matcher);

/* words must equal roost_grammar_mask_words. After the matcher stops, only the stop ids are allowed. */
int32_t roost_grammar_matcher_compute_mask(RoostGrammarMatcher* matcher, uint32_t* dest, size_t words);

/* A stop id after the matcher has stopped is accepted and changes nothing. */
int32_t roost_grammar_matcher_consume(RoostGrammarMatcher* matcher, uint32_t token);

/* 1 or 0, or a negative status. */
int32_t roost_grammar_matcher_is_accepting(RoostGrammarMatcher* matcher);
int32_t roost_grammar_matcher_is_stopped(const RoostGrammarMatcher* matcher);

#ifdef __cplusplus
}
#endif

#endif /* ROOST_GRAMMAR_H_ */
