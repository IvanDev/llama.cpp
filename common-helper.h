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

LLAMA_API void *llm_init_sampling_context();
LLAMA_API void llm_free_sampling_context(void* ctx);
int32_t llm_sampling_sample(void* samplingContext, void *llamaContext, int idx);
void llm_sampling_accept(void *samplingContext, void *llamaContext, int32_t id, bool applyGrammar);
void llm_sampling_reset(void *samplingContext);

#ifdef  __cplusplus
}
#endif


//#endif  common_helper_hpp 
