# PPMN: Pixel-Phrase Matching Network for One-Stage Panoptic Narrative Grounding

![](figures/framework.png)

> **PPMN: Pixel-Phrase Matching Network for One-Stage Panoptic Narrative Grounding**, <br>
> [Zihan Ding*](https://scholar.google.com/citations?user=czvpD10AAAAJ&hl=zh-TW), Zi-han Ding*, [Tianrui Hui](https://scholar.google.com/citations?user=ArjkrTkAAAAJ&hl=zh-TW), [Junshi Huang](https://scholar.google.com.sg/citations?user=FFB6lzQAAAAJ&hl=en), Xiaoming Wei, Xiaolin Wei and [Si Liu](https://scholar.google.com/citations?user=-QtVtNEAAAAJ&hl=en) <br>
> *ACM MM 2022 ([arxiv 2208.05647](https://arxiv.org/abs/2208.05647))*

## News

* [2022-08-12] Code will come soon.

## Abstract

Panoptic Narrative Grounding (PNG) is an emerging task whose goal is to segment visual objects of things and stuff categories described by dense narrative captions of a still image. The previous two-stage approach first extracts segmentation region proposals by an off-the-shelf panoptic segmentation model, then conducts coarse region-phrase matching to ground the candidate regions for each noun phrase. However, the two-stage pipeline usually suffers from the performance limitation of low-quality proposals in the first stage and the loss of spatial details caused by region feature pooling, as well as complicated strategies designed for things and stuff categories separately. To alleviate these drawbacks, we propose a one-stage end-to-end Pixel-Phrase Matching Network (PPMN), which directly matches each phrase to its corresponding pixels instead of region proposals and outputs panoptic segmentation by simple combination. Thus, our model can exploit sufficient and finer cross-modal semantic correspondence from the supervision of densely annotated pixel-phrase pairs rather than sparse region-phrase pairs. In addition, we also propose a Language-Compatible Pixel Aggregation (LCPA) module to further enhance the discriminative ability of phrase features through multi-round refinement, which selects the most compatible pixels for each phrase to adaptively aggregate the corresponding visual context. Extensive experiments show that our method achieves new state-of-the-art performance on the PNG benchmark with 4.0 absolute Average Recall gains.

## Citation

```
@inproceedings{Ding_2022_ACMMM,
    author    = {Ding, Zihan and Ding, Zi-han and Hui, Tianrui and Huang, Junshi and Wei, Xiaoming and Wei, Xiaolin and Liu, Si},
    title     = {PPMN: Pixel-Phrase Matching Network for One-Stage Panoptic Narrative Grounding},
    booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
    year      = {2022}
```