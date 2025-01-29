//
//  common-helper.hpp
//  
//
//  Created by Ivan Isaev on 28/05/2024.
//
//#ifndef common_helper_hpp
//#define common_helper_hpp
#pragma once

#include "llama.h"


#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct llm_sampling_params {
//    int32_t     n_prev;
//    int32_t     n_probs;
//    int32_t     min_keep;
//    int32_t     top_k;
//    float       top_p;
//    float       min_p;
//    float       tfs_z;
//    float       typical_p;
//    float       temp;
//    float       dynatemp_range;
//    float       dynatemp_exponent;
//    int32_t     penalty_last_n;
//    float       penalty_repeat;
//    float       penalty_freq;
//    float       penalty_present;
//    int32_t     mirostat;
//    float       mirostat_tau;
//    float       mirostat_eta;
//    bool        penalize_nl;
    
    int32_t n_prev;    // number of previous tokens to remember
    int32_t n_probs;     // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t min_keep;     // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t top_k;    // <= 0 to use vocab size
    float   top_p; // 1.0 = disabled
    float   min_p; // 0.0 = disabled
    float   xtc_probability; // 0.0 = disabled
    float   xtc_threshold; // > 0.5 disables XTC
    float   typ_p; // typical_p, 1.0 = disabled
    float   temp; // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float   dynatemp_range; // 0.0 = disabled
    float   dynatemp_exponent; // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t penalty_last_n;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   penalty_repeat; // 1.0 = disabled
    float   penalty_freq; // 0.0 = disabled
    float   penalty_present; // 0.0 = disabled
    float   dry_multiplier;  // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
    float   dry_base; // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
    int32_t dry_allowed_length;     // tokens extending repetitions beyond this receive penalty
    int32_t dry_penalty_last_n;    // how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
    int32_t mirostat;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau; // target entropy
    float   mirostat_eta; // learning rate
    bool    penalize_nl; // consider newlines as a repeatable token
    bool    ignore_eos;
    bool    no_perf;
    
    uint32_t    seed;
} llm_sampling_params;

LLAMA_API void *llm_init_sampling_context(const struct llama_model *model, llm_sampling_params parameters);
LLAMA_API void llm_free_sampling_context(void* ctx);
int32_t llm_sampling_sample(void* samplingContext, struct llama_context *llamaContext, int idx);
void llm_sampling_accept(void *samplingContext, struct llama_context *llamaContext, int32_t id, bool applyGrammar);
void llm_sampling_reset(void *samplingContext);
const char *llm_model_info(const char *model_file_name);

#ifdef  __cplusplus
}
#endif


//#endif  common_helper_hpp 
