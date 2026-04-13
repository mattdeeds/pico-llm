#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>

typedef struct {
    char **vocab;       // token id -> string
    int vocab_size;
} Tokenizer;

void tokenizer_init(Tokenizer *t, uint8_t *vocab_data, int vocab_size);
void tokenizer_free(Tokenizer *t);
int tokenizer_encode(const Tokenizer *t, const char *text, int *tokens, int max_tokens);
const char *tokenizer_decode(const Tokenizer *t, int token);

#endif
