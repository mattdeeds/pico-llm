#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>

void tokenizer_init(Tokenizer *t, uint8_t *vocab_data, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char *));

    // Parse vocab data: each entry is a length-prefixed string
    uint8_t *ptr = vocab_data;
    for (int i = 0; i < vocab_size; i++) {
        uint16_t len = ptr[0] | (ptr[1] << 8);
        ptr += 2;
        t->vocab[i] = malloc(len + 1);
        memcpy(t->vocab[i], ptr, len);
        t->vocab[i][len] = '\0';
        ptr += len;
    }
}

void tokenizer_free(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
}

// Convert ASCII text to byte-level BPE representation.
// Space → Ġ (U+0120, UTF-8: 0xC4 0xA0), newline → Ċ (U+010A, UTF-8: 0xC4 0x8A).
// Printable ASCII passes through unchanged.
static int ascii_to_bpe(const char *text, char *out, int max_len) {
    int j = 0;
    for (int i = 0; text[i] && j < max_len - 2; i++) {
        uint8_t c = (uint8_t)text[i];
        if (c == ' ') {
            out[j++] = (char)0xC4;
            out[j++] = (char)0xA0;
        } else if (c == '\n') {
            out[j++] = (char)0xC4;
            out[j++] = (char)0x8A;
        } else if (c == '\t') {
            out[j++] = (char)0xC4;
            out[j++] = (char)0x89;
        } else {
            out[j++] = (char)c;
        }
    }
    out[j] = '\0';
    return j;
}

// Greedy longest-match tokenizer. Not true BPE (which merges bottom-up),
// but produces reasonable tokenization for common text.
int tokenizer_encode(const Tokenizer *t, const char *text, int *tokens, int max_tokens) {
    // Convert to BPE byte-level representation
    char bpe[512];
    int bpe_len = ascii_to_bpe(text, bpe, sizeof(bpe));

    int n = 0;
    int pos = 0;

    while (pos < bpe_len && n < max_tokens) {
        int best_len = 0;
        int best_id = -1;

        for (int i = 0; i < t->vocab_size; i++) {
            int tlen = (int)strlen(t->vocab[i]);
            if (tlen > best_len && tlen <= bpe_len - pos) {
                if (memcmp(t->vocab[i], bpe + pos, tlen) == 0) {
                    best_len = tlen;
                    best_id = i;
                }
            }
        }

        if (best_id >= 0) {
            tokens[n++] = best_id;
            pos += best_len;
        } else {
            pos++; // skip unmatchable byte
        }
    }

    return n;
}

const char *tokenizer_decode(const Tokenizer *t, int token) {
    if (token < 0 || token >= t->vocab_size) return "";
    return t->vocab[token];
}
