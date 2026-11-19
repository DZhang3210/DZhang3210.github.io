---
title: Address matching: one missing house number explains most of the gap
date: November 2026
description: Tree models beat neural networks at telling real address matches from look-alike streets. Almost all of that lead comes from pairs where one side has no house number.
draft: true
---

<p class="lede">I put classical tree models and four neural networks on the same set of real address pairs and scored them by the kind of pair, not one overall number. The neural models aren't worse at streets. They're worse at one thing: a house number missing from one side.</p>

## The setup

Address matching is the step in geocoding where you decide whether two address strings are the same place. `12 ELM ST` and `12 ELM STREET`: yes. `431 HALLETT ST` and `431 HARRIET ST`: no, even though the letters are close.

I built pairs from two open datasets that describe the same places in Connecticut, [Overture Maps](https://overturemaps.org/) and [Foursquare Open Source Places](https://opensource.foursquare.com/os-places/). Overture's own matching between them gives the true pairs. For the non-matches I built several kinds: random pairs (easy), same street with another house number, and look-alike streets, made by swapping a real street for a similar-looking real street in the same ZIP code. The test set is made of ZIP codes no model trained on.

Then I compared:

- **Classical models** (random forest, XGBoost, logistic regression) that get the address pre-split into house number, street and ZIP, with string-similarity scores for each part;
- **Neural models** that read only the two raw strings: a character-level siamese BiLSTM, ESIM, and DistilBERT as a bi-encoder (each address encoded separately) and as a cross-encoder (both read together).

## The result

All models handle the easy cases: identical addresses, a different number on the same street, random pairs. The hard question is separating real matches whose text differs from fake look-alike streets. On that, ranked by AUC (the chance a real match scores above a fake), the tree models lead:

{{chart:auc_overall}}

But split the real matches by what happened to their house number, and the picture changes:

{{chart:auc_groups}}

With the same number on both sides, the transformers are as good as the random forest. With the number missing from one side (`641 SCOFIELDTOWN RD` / `SCOFIELDTOWN RD`), the bi-encoder is at or below chance, while the random forest, which sees the house number as its own field, still does well. The cross-encoder's whole advantage over the bi-encoder comes from these pairs.

Part of that group can't be solved at all: through a mistake in how I built one kind of non-match, some different-place pairs look exactly like `74 BEACH RD` / `BEACH RD` too. Every model is close to chance on separating those from real matches.

## The takeaway

Scoring by pair type is what showed this. One overall number would have said "trees beat transformers". The real story is narrower and fixable: give the neural models the house number separately, or a "number missing" signal.

**More:** [the paper (PDF)](https://github.com/DZhang3210/address-matching-by-pair-type/blob/main/paper/address-matching-by-pair-type.pdf) · [code](https://github.com/DZhang3210/address-matching-by-pair-type) · [models](https://huggingface.co/zhangdav/address-matching-by-pair-type) · [dataset](https://huggingface.co/datasets/zhangdav/address-matching-by-pair-type)
