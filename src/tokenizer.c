#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>

// TODO: Implement BPE tokenizer
// Vocab will be loaded from the SD card weight file header

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

int tokenizer_encode(Tokenizer *t, const char *text, int *tokens, int max_tokens) {
    (void)t;
    (void)text;
    (void)tokens;
    (void)max_tokens;
    // TODO: BPE encoding
    return 0;
}

const char *tokenizer_decode(const Tokenizer *t, int token) {
    if (token < 0 || token >= t->vocab_size) return "";
    return t->vocab[token];
}
