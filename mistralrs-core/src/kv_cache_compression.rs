use core::f32;

use candle_core::{bail, Device, Result, Tensor};
use candle_core::{IndexOp, D};
use candle_nn::ops::{softmax, softmax_last_dim};

use crate::layers_masker::{apply_tril, masked_fill};
use crate::ops::TopKLastDimOp;
pub(crate) enum KVCacheCompressionMethod {
    SnapKV,
}

pub(crate) enum KVCacheCompressionPoolingMethod {
    Max,
    Avg,
}

pub struct KVCacheCompressionConfig {
    pub(crate) method: KVCacheCompressionMethod,
    pub(crate) window_size: usize,
    pub(crate) max_capacity_prompt: usize,
    pub(crate) kernel_size: usize,
    pub(crate) pooling: KVCacheCompressionPoolingMethod,
}

impl KVCacheCompressionConfig {
    pub fn new(
        method: KVCacheCompressionMethod,
        window_size: usize,
        max_capacity_prompt: usize,
        kernel_size: usize,
        pooling: KVCacheCompressionPoolingMethod,
    ) -> Result<Self> {
        if max_capacity_prompt <= window_size {
            bail!("max_capacity_prompt must be greater than window_size");
        }
        Ok(Self {
            method,
            window_size,
            max_capacity_prompt,
            kernel_size,
            pooling,
        })
    }
}

fn snap_kv(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    config: &KVCacheCompressionConfig,
) -> Result<(Tensor, Tensor)> {
    let (_, _, q_len, head_dim) = query.dims4()?;
    if q_len < config.max_capacity_prompt {
        return Ok((query.clone(), key.clone()));
    }
    let selected_q = query.i((.., .., q_len - config.window_size.., ..))?;
    let mut attn_weights = selected_q.matmul(&key.transpose(2, 3)?)?; //[batch_size, num_heads,config.window_size,q_len]
    attn_weights = (attn_weights / (head_dim as f64).sqrt())?;
    let (batch_size, num_heads, _, _) = attn_weights.dims4()?;
    //down triangle mask in config.window_size
    let mut attention_mask = Tensor::full(
        f32::NEG_INFINITY,
        (config.window_size, config.window_size),
        query.device(),
    )?;
    let tri_mask =
        apply_tril(&attention_mask.ones_like().unwrap(), 0)?.to_dtype(candle_core::DType::U8)?;
    attention_mask = masked_fill(&attention_mask, &tri_mask, 0f32)?.expand((
        batch_size,
        num_heads,
        config.window_size,
        config.window_size,
    ))?;
    //add mask
    let indices = Tensor::arange(
        (q_len - config.window_size) as u32,
        q_len as u32,
        &query.device(),
    )?;
    attn_weights = attn_weights.index_add(&indices, &attention_mask, 3)?; //[batch_size, num_heads,config.window_size,q_len]

    attn_weights = softmax_last_dim(&attn_weights)?;
    attn_weights = attn_weights.i((.., .., .., ..q_len - config.window_size))?;
    attn_weights = attn_weights.sum(2)?; //[batch_size,num_heads,config.window_size,q_len-config.window_size]

    attn_weights =
        attn_weights.pad_with_zeros(2, config.kernel_size / 2, config.kernel_size / 2)?;
    attn_weights = attn_weights.unsqueeze(D::Minus1)?; // candle does not have pool1d, so we need to unsqueeze the last dim

    attn_weights = match config.pooling {
        KVCacheCompressionPoolingMethod::Max => attn_weights.max_pool2d((config.kernel_size, 1))?,
        KVCacheCompressionPoolingMethod::Avg => attn_weights.avg_pool2d((config.kernel_size, 1))?,
    };
    attn_weights = attn_weights.squeeze(D::Minus1)?;
    let mut indices = attn_weights
        .topk(config.max_capacity_prompt - config.window_size)?
        .indices;
    let (batch_size, num_heads, windows_size, _) = indices.dims4()?;
    indices =
        indices
            .unsqueeze(D::Minus1)?
            .expand((batch_size, num_heads, windows_size, head_dim))?;
    let k_past_compress = key
        .i((.., .., ..q_len - config.window_size, ..))?
        .gather(&indices, 2)?;
    let v_past_compress = value
        .i((.., .., ..q_len - config.window_size, ..))?
        .gather(&indices, 2)?;
    let k_cur = key.i((.., .., q_len - config.window_size.., ..))?;
    let v_cur = value.i((.., .., q_len - config.window_size.., ..))?;
    let k = Tensor::cat(&[k_past_compress, k_cur], 2)?;
    let v = Tensor::cat(&[v_past_compress, v_cur], 2)?;
    Ok((k, v))
}

pub(crate) fn compress_kv_cache(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    config: &KVCacheCompressionConfig,
) -> Result<(Tensor, Tensor)> {
    match config.method {
        KVCacheCompressionMethod::SnapKV => snap_kv(query, key, value, config),
    }
}

#[test]
fn test_apply_tril() {
    use crate::layers_masker::apply_tril;
    let x = Tensor::full(f32::NEG_INFINITY, (5, 5), &Device::Cpu).unwrap();
    //let x = Tensor::zeros((5,5),candle_core::DType::F32,&Device::Cpu).unwrap();
    let mask = apply_tril(&x.ones_like().unwrap(), 0)
        .unwrap()
        .to_dtype(candle_core::DType::U8)
        .unwrap();
    println!("{}", mask);
    let y = masked_fill(&x, &mask, 0f32).unwrap();
    println!("{}", y);
}
