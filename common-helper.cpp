//
//  common-helper.cpp
//  
//
//  Created by Ivan Isaev on 28/05/2024.
//

#include "common-helper.h"
#include "common.h"
#include "sampling.h"

common_sampler_params getParams(llm_sampling_params params) {
    common_sampler_params result = common_sampler_params();
    
    result.n_prev = params.n_prev;
    result.n_probs = params.n_probs;
    result.min_keep = params.min_keep;
    result.top_k = params.top_k;
    result.top_p = params.top_p;
    result.min_p = params.min_p;
//    result.tfs_z = params.tfs_z;
    result.typ_p = params.typ_p;
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

void *llm_init_sampling_context(const struct llama_model *model, llm_sampling_params parameters) {
    common_sampler_params params = getParams(parameters);
    params.print();
//    llama_sampling_print(params);
//    llama_sampling_order_print(params);
    
    struct common_sampler * ctx_sampling = common_sampler_init(model, params);
    return ctx_sampling;
}

void llm_free_sampling_context(void* ctx) {
    common_sampler_free((struct common_sampler*) ctx);
}

int32_t llm_sampling_sample(void* samplingContext, struct llama_context *llamaContext, int idx = -1) {
    return common_sampler_sample((struct common_sampler *)samplingContext, (struct llama_context*)llamaContext, idx);
//    return llama_sampling_sample((struct llama_sampling_context*)samplingContext, (struct llama_context*)llamaContext, NULL, idx);
}

void llm_sampling_accept(void *samplingContext, struct llama_context *llamaContext, int32_t id, bool applyGrammar) {
    return common_sampler_accept((struct common_sampler*)samplingContext, id, applyGrammar);
//    llama_sampling_accept((struct llama_sampling_context*)samplingContext, (struct llama_context*)llamaContext, id, applyGrammar);
}

void llm_sampling_reset(void *samplingContext) {
    common_sampler_reset((struct common_sampler*)samplingContext);
//    llama_sampling_reset((struct llama_sampling_context*)samplingContext);
}

