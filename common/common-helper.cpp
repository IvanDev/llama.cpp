//
//  common-helper.cpp
//  
//
//  Created by Ivan Isaev on 28/05/2024.
//

#include "common-helper.h"
#include "common.h"
#include "sampling.h"
#include "llama-model-loader.h"
#include "llama-model.h"


common_params_sampling getParams(llm_sampling_params params) {
    common_params_sampling result = common_params_sampling();
    
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
    // result.penalize_nl = params.penalize_nl;
    result.seed = params.seed;
    
    return result;
}

void *llm_init_sampling_context(const struct llama_model *model, llm_sampling_params parameters) {
    common_params_sampling params = getParams(parameters);
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

const std::string val_array(std::string prefix, std::array<uint32_t, LLAMA_MAX_LAYERS> v, int max) {
    std::string result = "";
    for (int i = 0; i < max; i++) {
        result += prefix + std::to_string(i) + "\": " + std::to_string(v[i]) + ", ";
    }
    return result;
}

const char *llm_model_info(const char *model_file_name) {
    const std::string fname = model_file_name;
    auto print_f = [](const std::function<uint32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<uint32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "\"[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]\"";
        } else {
            ss << v[0];
        }

        return std::string(ss.str());
    };

    struct llama_model_params model_params = llama_model_default_params();
    std::string result = "{ ";
    llama_model * model = new llama_model(model_params);
    try {
        std::vector<std::string> splits = {};
        llama_model_loader ml(fname, splits, model_params.use_mmap, model_params.check_tensors, model_params.kv_overrides);
        result += "\"model\": {";
        result += "\"file_format\": \"" + std::string(llama_file_version_name(ml.fver)) + "\",";
        result += "\"file_type\": \"" + std::string(ml.ftype_name()) + "\",";
        result += "\"file_size\": " + std::to_string(ml.n_bytes);
        result += "},";

        try {
            model->load_arch(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            model->load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            model->load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model->load_stats(ml);
        result += "\"name\": \"" + model->name + "\",";
        //Model params:

        result += "\"description\": \"" + model->desc() + "\",";
//        LLAMA_LOG_INFO("%s: n_head           = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer).c_str());

        if (!model->hparams.vocab_only) {
//            const char * rope_scaling_type = LLAMA_ROPE_SCALING_TYPES.at(model->hparams.rope_scaling_type_train);
            result += "\"hparams\": {";
            result += "\"n_ctx_train\": " + std::to_string(model->hparams.n_ctx_train) + ",";
            result += "\"n_embd\": " + std::to_string(model->hparams.n_embd) + ",";
            result += "\"n_layer\": " + std::to_string(model->hparams.n_layer) + ",";
//            result += val_array("\"n_head", model->hparams.n_head, model->hparams.n_layer);
            result += "\"n_head\": " + std::string(print_f([&](uint32_t il) { return model->hparams.n_head(il); }, model->hparams.n_layer).c_str()) + ",";
            result += "\"n_head_kv\": " + print_f([&](uint32_t il) { return model->hparams.n_head_kv(il); }, model->hparams.n_layer) + ",";
            result += "\"n_rot\": " + std::to_string(model->hparams.n_rot) + ",";
            result += "\"n_swa\": " + std::to_string(model->hparams.n_swa) + ",";
            result += "\"n_embd_head_k\": " + std::to_string(model->hparams.n_embd_head_k) + ",";
            result += "\"n_embd_head_v\": " + std::to_string(model->hparams.n_embd_head_v) + ",";
            result += "\"n_gqa\": " + print_f([&](uint32_t il) { return model->hparams.n_gqa(il); }, model->hparams.n_layer) + ",";
            result += "\"n_embd_k_gqa\": " + print_f([&](uint32_t il) { return model->hparams.n_embd_k_gqa(il); }, model->hparams.n_layer) + ",";
            result += "\"n_embd_v_gqa\": " + print_f([&](uint32_t il) { return model->hparams.n_embd_v_gqa(il); }, model->hparams.n_layer) + ",";
            result += "\"f_norm_eps\": " + std::to_string(model->hparams.f_norm_eps) + ",";
            result += "\"f_norm_rms_eps\": " + std::to_string(model->hparams.f_norm_rms_eps) + ",";
            result += "\"f_clamp_kqv\": " + std::to_string(model->hparams.f_clamp_kqv) + ",";
            result += "\"f_max_alibi_bias\": " + std::to_string(model->hparams.f_max_alibi_bias) + ",";
            result += "\"f_logit_scale\": " + std::to_string(model->hparams.f_logit_scale) + ",";
            result += "\"n_ff\": " + print_f([&](uint32_t il) { return model->hparams.n_ff(il); }, model->hparams.n_layer) + ",";
            result += "\"n_expert\": " + std::to_string(model->hparams.n_expert) + ",";
            result += "\"n_expert_used\": " + std::to_string(model->hparams.n_expert_used) + ",";
            result += "\"causal_attn\": " + std::to_string(model->hparams.causal_attn) + ",";
            result += "\"pooling_type\": " + std::to_string(model->hparams.pooling_type) + ",";
            result += "\"rope_type\": " + std::to_string(model->hparams.rope_type) + ",";
            result += "\"rope_scaling_type\": " + std::to_string(model->hparams.rope_scaling_type_train) + ",";
            result += "\"rope_freq_base_train\": " + std::to_string(model->hparams.rope_freq_base_train) + ",";
            result += "\"rope_freq_scale_train\": " + std::to_string(model->hparams.rope_freq_scale_train) + ",";
            result += "\"n_ctx_orig_yarn\": " + std::to_string(model->hparams.n_ctx_orig_yarn) + ",";
            result += "\"rope_finetuned\": " + std::to_string(model->hparams.rope_finetuned) + ",";
            result += "\"ssm_d_conv\": " + std::to_string(model->hparams.ssm_d_conv) + ",";
            result += "\"ssm_d_inner\": " + std::to_string(model->hparams.ssm_d_inner) + ",";
            result += "\"ssm_d_state\": " + std::to_string(model->hparams.ssm_d_state) + ",";
            result += "\"ssm_dt_rank\": " + std::to_string(model->hparams.ssm_dt_rank) + ",";
            result += "\"ssm_dt_b_c_rms\": " + std::to_string(model->hparams.ssm_dt_b_c_rms) + ",";
            if (result.back() == ',') {
                result.pop_back();
            }
            result += "},";
        }

        result += "\"stats\": {";
        result += "\"model_type\": \"" + model->type_name() + "\",";
        size_t n_elements = model->n_elements();
        if (n_elements >= 1e12) {
            result += "\"model_params\": \"" + std::to_string(n_elements*1e-12) + "T\",";
        } else if (n_elements >= 1e9) {
            result += "\"model_params\": \"" + std::to_string(n_elements*1e-9) + "B\",";
        } else if (n_elements >= 1e6) {
            result += "\"model_params\": \"" + std::to_string(n_elements*1e-6) + "M\",";
        } else {
            result += "\"model_params\": \"" + std::to_string(n_elements*1e-3) + "K\",";
        }


        if (model->arch == LLM_ARCH_DEEPSEEK) {
            result += "\"n_layer_dense_lead\": " + std::to_string(model->hparams.n_layer_dense_lead) + ",";
            result += "\"n_ff_exp\": " + std::to_string(model->hparams.n_ff_exp) + ",";
            result += "\"n_expert_shared\": " + std::to_string(model->hparams.n_expert_shared) + ",";
            result += "\"expert_weights_scale\": " + std::to_string(model->hparams.expert_weights_scale) + ",";
        }

        if (model->arch == LLM_ARCH_DEEPSEEK2) {
            result += "\"n_layer_dense_lead\": " + std::to_string(model->hparams.n_layer_dense_lead) + ",";
            result += "\"n_lora_q\": " + std::to_string(model->hparams.n_lora_q) + ",";
            result += "\"n_lora_kv\": " + std::to_string(model->hparams.n_lora_kv) + ",";
            result += "\"n_ff_exp\": " + std::to_string(model->hparams.n_ff_exp) + ",";
            result += "\"n_expert_shared\": " + std::to_string(model->hparams.n_expert_shared) + ",";
            result += "\"expert_weights_scale\": " + std::to_string(model->hparams.expert_weights_scale) + ",";
            result += "\"expert_weights_norm\": " + std::to_string(model->hparams.expert_weights_norm) + ",";
//            result += "\"expert_gating_func\": \"" + std::string( llama_expert_gating_func_type((enum llama_expert_gating_func_type) model->hparams.expert_gating_func) ) + "\",";
            result += "\"rope_yarn_log_mul\": " + std::to_string(model->hparams.rope_yarn_log_mul) + ",";
        }

        if (model->arch == LLM_ARCH_QWEN2MOE) {
            result += "\"n_ff_exp\": " + std::to_string(model->hparams.n_ff_exp) + ",";
            result += "\"n_ff_shexp\": " + std::to_string(model->hparams.n_ff_shexp) + ",";
        }

        if (model->arch == LLM_ARCH_MINICPM || model->arch == LLM_ARCH_GRANITE || model->arch == LLM_ARCH_GRANITE_MOE) {
            result += "\"f_embedding_scale\": " + std::to_string(model->hparams.f_embedding_scale) + ",";
            result += "\"f_residual_scale\": " + std::to_string(model->hparams.f_residual_scale) + ",";
            result += "\"f_attention_scale\": " + std::to_string(model->hparams.f_attention_scale) + ",";
        }
        if (result.back() == ',') {
            result.pop_back();
        }
        result += "},";// stats
        //vocab
        const llama_vocab & vocab = model->vocab;
        result += "\"vocab\": {";
        result += "\"vocab_type\": \"" + vocab.type_name() + "\",";
        result += "\"n_vocab\": " + std::to_string(vocab.n_tokens());
//        result += "\"n_merges\": " + std::to_string((uint32_t)vocab.bpe_ranks.size()) + ",";
//        LLAMA_LOG_INFO("%s: vocab type       = %s\n",     __func__, type_name().c_str());
//        LLAMA_LOG_INFO("%s: n_vocab          = %u\n",     __func__, vocab.n_tokens());
//        LLAMA_LOG_INFO("%s: n_merges         = %u\n",     __func__, (uint32_t) bpe_ranks.size());
        result += "},";

        result += "\"tokens\": {";
        if (vocab.token_bos() != LLAMA_TOKEN_NULL) {
            result += "\"BOS\": \"" + std::string(vocab.token_get_text(vocab.token_bos())) + "\",";
        }
        if (vocab.token_eos() != LLAMA_TOKEN_NULL) {
            result += "\"EOS\": \"" + std::string(vocab.token_get_text(vocab.token_eos())) + "\",";
        }
        if (vocab.token_eot() != LLAMA_TOKEN_NULL) {
            result += "\"EOT\": \"" + std::string(vocab.token_get_text(vocab.token_eot())) + "\",";
        }
        if (vocab.token_eom() != LLAMA_TOKEN_NULL) {
            result += "\"EOM\": \"" + std::string(vocab.token_get_text(vocab.token_eom())) + "\",";
        }
        if (vocab.token_unk() != LLAMA_TOKEN_NULL) {
            result += "\"UNK\": \"" + std::string(vocab.token_get_text(vocab.token_unk())) + "\",";
        }
        if (vocab.token_sep() != LLAMA_TOKEN_NULL) {
            result += "\"SEP\": \"" + std::string(vocab.token_get_text(vocab.token_sep())) + "\",";
        }
        if (vocab.token_pad() != LLAMA_TOKEN_NULL) {
            result += "\"PAD\": \"" + std::string(vocab.token_get_text(vocab.token_pad())) + "\",";
        }
        if (vocab.token_nl() != LLAMA_TOKEN_NULL) {
            result += "\"LF\": \"" + std::string(vocab.token_get_text(vocab.token_nl())) + "\",";
        }
        if (vocab.token_prefix() != LLAMA_TOKEN_NULL) {
            result += "\"FIM_PRE\": \"" + std::string(vocab.token_get_text(vocab.token_prefix())) + "\",";
        }
        if (vocab.token_suffix() != LLAMA_TOKEN_NULL) {
            result += "\"FIM_SUF\": \"" + std::string(vocab.token_get_text(vocab.token_suffix())) + "\",";
        }
        if (vocab.token_middle() != LLAMA_TOKEN_NULL) {
            result += "\"FIM_MID\": \"" + std::string(vocab.token_get_text(vocab.token_middle())) + "\",";
        }
        result += "\"max_token_len\": " + std::to_string(vocab.max_token_len()) + ",";
        if (result.back() == ',') {
            result.pop_back();
        }
        result += "}";

    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        delete model;
        return "{}";
    }

    delete model;
    result += "}";
    return strdup(result.c_str());
}
