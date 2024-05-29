//
//  common-helper.cpp
//  
//
//  Created by Ivan Isaev on 28/05/2024.
//

#include "common-helper.h"
#include "common.h"
#include "sampling.h"

void *llm_init_sampling_context() {
    llama_sampling_params params = llama_sampling_params();
    params.penalty_repeat = 1.1;
    params.seed = 1234;
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params);
    return ctx_sampling;
}

void llm_free_sampling_context(void* ctx) {
    llama_sampling_free((struct llama_sampling_context*) ctx);
}

int32_t llm_sampling_sample(void* samplingContext, void *llamaContext, int idx) {
    return llama_sampling_sample((struct llama_sampling_context*)samplingContext, (struct llama_context*)llamaContext, NULL, idx);
}

void llm_sampling_accept(void *samplingContext, void *llamaContext, int32_t id, bool applyGrammar) {
    llama_sampling_accept((struct llama_sampling_context*)samplingContext, (struct llama_context*)llamaContext, id, applyGrammar);
}

void llm_sampling_reset(void *samplingContext) {
    llama_sampling_reset((struct llama_sampling_context*)samplingContext);
}

