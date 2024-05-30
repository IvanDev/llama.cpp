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
    int32_t     n_prev;
    int32_t     n_probs;
    int32_t     min_keep;
    int32_t     top_k;
    float       top_p;
    float       min_p;
    float       tfs_z;
    float       typical_p;
    float       temp;
    float       dynatemp_range;
    float       dynatemp_exponent;
    int32_t     penalty_last_n;
    float       penalty_repeat;
    float       penalty_freq;
    float       penalty_present;
    int32_t     mirostat;
    float       mirostat_tau;
    float       mirostat_eta;
    bool        penalize_nl;
    uint32_t    seed;
} llm_sampling_params;

LLAMA_API void *llm_init_sampling_context(llm_sampling_params parameters);
LLAMA_API void llm_free_sampling_context(void* ctx);
int32_t llm_sampling_sample(void* samplingContext, void *llamaContext, int idx);
void llm_sampling_accept(void *samplingContext, void *llamaContext, int32_t id, bool applyGrammar);
void llm_sampling_reset(void *samplingContext);

#ifdef  __cplusplus
}
#endif


//#endif  common_helper_hpp 
