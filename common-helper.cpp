//
//  common-helper.cpp
//  
//
//  Created by Ivan Isaev on 28/05/2024.
//

#include "common-helper.h"
#include "common.h"
#include "sampling.h"

llama_sampling_params getParams(llm_sampling_params params) {
    llama_sampling_params result = llama_sampling_params();
    
    result.n_prev = params.n_prev;
    result.n_probs = params.n_probs;
    result.min_keep = params.min_keep;
    result.top_k = params.top_k;
    result.top_p = params.top_p;
    result.min_p = params.min_p;
    result.tfs_z = params.tfs_z;
    result.typical_p = params.typical_p;
    result.temp = params.temp;
    result.dynatemp_range = params.dynatemp_range;
    result.dynatemp_exponent = params.dynatemp_exponent;
    result.penalty_last_n = params.penalty_last_n;
    result.penalty_repeat = params.penalty_repeat;
    result.penalty_freq = params.penalty_freq;
    result.penalty_present = params.penalty_present;
    result.mirostat = params.mirostat;
    result.mirostat_tau = params.mirostat_tau;
    result.mirostat_eta = params.mirostat_eta;
    result.penalize_nl = params.penalize_nl;
    result.seed = params.seed;
    
    return result;
}

void *llm_init_sampling_context(llm_sampling_params parameters) {
    llama_sampling_params params = getParams(parameters);
    
    llama_sampling_print(params);
    llama_sampling_order_print(params);
    
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

