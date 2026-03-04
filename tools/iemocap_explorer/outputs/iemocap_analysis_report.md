# IEMOCAP Dataset Analysis Report
## For Echo-Mind System: Mamba-Transformer Hybrid Multimodal Emotion Recognition

---

## 0. Dataset Overview — What is IEMOCAP?

IEMOCAP (Interactive Emotional Dyadic Motion Capture) is the **most widely used benchmark** in multimodal emotion recognition research, created by USC in 2008. It contains **10,039 utterances** from 10 actors (5 male, 5 female) across 5 recording sessions. Each session features one male-female pair performing both improvised and scripted emotional dialogues.

Every utterance has: audio (.wav), transcript (text), emotion categorical labels (annotated by 3+ evaluators), and continuous Valence-Arousal-Dominance (V-A-D) dimensional scores (annotated by 2+ evaluators). This makes it ideal for your multimodal (text + audio) research in Q1–Q2.

**Key numbers from your parsed data:**
- Total utterances: **10,039**
- Total audio: **12.4 hours**
- 100% of samples have both audio and text (perfect completeness)
- 5 sessions → 5-fold Leave-One-Session-Out (LOSO) cross-validation

---

## 1. Raw Emotion Distribution (Plot 01)

**What it shows:** All 11 emotion categories in IEMOCAP with their frequency counts.

**Key findings:**

- **"No Agreement" is the largest category (2,507 / 25%)** — This means a quarter of all utterances had no majority consensus among the 3+ annotators. This is NOT a data quality problem — it reflects the genuine ambiguity of human emotion. These samples are typically excluded from classification experiments but are **gold for your soft-label probability matrix** approach, because they inherently carry mixed-emotion signals.

- **Frustrated (1,849) is dominant among labeled emotions** — This is an artifact of the acted scenarios in IEMOCAP, which include many conflict-based scripts.

- **Long tail problem:** Surprised (107), Fearful (40), Disgusted (2), Other (3) are too rare for reliable classification. No paper uses these categories — they're always dropped.

- **Happy is surprisingly small (595)** — Pure "happy" is rare because many happy-sounding utterances were labeled "excited" instead by annotators. This is why the literature always merges Happy + Excited.

**Design decision:**
→ **Drop** No Agreement, Surprised, Fearful, Disgusted, Other from classification training
→ For your soft-label approach: **keep** No Agreement samples and use the individual evaluator labels as the probability distribution — these are your most interesting training samples because they demonstrate true emotional ambiguity

---

## 2. Standard 4-Class Distribution (Plot 02)

**What it shows:** The standard benchmark setting used in 90%+ of IEMOCAP papers: Angry, Happy (merged with Excited), Sad, Neutral.

**Key findings:**

- **5,531 usable utterances** (55% of total) after filtering
- **Imbalance ratio: 1.58x** (Neutral 1,708 vs Sad 1,084) — this is moderate. Many datasets have 5-10x imbalance, so IEMOCAP 4-class is relatively balanced.
- The pie chart shows a roughly even split: Neutral 30.9%, Happy 29.6%, Angry 19.9%, Sad 19.6%

**Design decisions:**
→ **Standard CrossEntropyLoss is acceptable** — the 1.58x imbalance is mild enough that weighted loss isn't strictly necessary, but using class weights will give you a small F1 boost on Angry/Sad
→ **Recommended class weights:** `[1.0, 0.67, 1.0, 1.02]` for [Angry, Happy+Excited, Neutral, Sad] (inverse frequency normalized)
→ **Always report both WA (Weighted Accuracy) and UA (Unweighted Accuracy)** — reviewers expect both for IEMOCAP

---

## 3. 6-Class Per Session (Plot 03) — LOSO Cross-Validation Planning

**What it shows:** How emotion categories are distributed across each recording session, which directly impacts your Leave-One-Session-Out (LOSO) evaluation protocol.

**Key findings — THIS IS CRITICAL:**

- **Session 4 has extreme Angry overrepresentation (~325)** compared to Session 2 (~140). When Session 4 is your test fold, the model will face a disproportionate amount of angry test samples.

- **Session 4 has very low Happy (~65)** — nearly half of other sessions. This means your LOSO fold 4 will have poor Happy recall simply due to scarce test data, causing high variance in that fold.

- **Session 5 has massive Frustrated (~480)** — the highest of any session, while Session 1 has only ~280. This creates uneven fold difficulty.

- **Excited distribution varies heavily:** Session 5 (~300) has double Session 1 (~140). Since Happy+Excited get merged in 4-class, this means "Happy" class has very different composition across folds.

**Design decisions:**
→ **Always report per-fold results** in addition to mean — if your model gets 80% on 4 folds but 65% on fold 4, the mean of 77% is misleading
→ **Use stratified batching within each training fold** — ensure each mini-batch roughly preserves the emotion distribution of its training set
→ **For quick experiments (not final paper), you can use Session 1-4 train / Session 5 test** as a single split to iterate faster. Final paper must use full 5-fold LOSO.

---

## 4. Audio Duration Distribution (Plot 04)

**What it shows:** How long each audio clip is, with key percentile markers for sequence length configuration.

**Key findings:**

- **Heavy right skew:** Most utterances are 1-5 seconds, with a long tail up to 34s
- **Mean: 4.5s, Median: 3.5s** — the mean is pulled up by outliers
- **P95: 10.5s** — 95% of all utterances are under 10.5 seconds
- **Box plots per session are consistent** — no one session has dramatically different utterance lengths
- Outliers (>15s) exist in every session but are rare

**Design decisions — THIS DIRECTLY SETS YOUR MAMBA SEQUENCE LENGTH:**

→ **Set `max_audio_length = 11 seconds`** (P95 rounded up). This covers 95% of data with no truncation.

→ At 16kHz sampling (standard for Wav2Vec2.0): 11s × 16000 = **176,000 raw samples**. After Wav2Vec2.0 feature extraction (320x downsample): **~550 feature frames**. This is your audio sequence length for Mamba.

→ **For comparison:** A Transformer with full self-attention on 550 tokens needs O(550²) = 302,500 attention computations. Mamba needs O(550) = 550. This is your paper's efficiency argument.

→ **Padding strategy:** Pad shorter clips to `max_length` with zeros, use attention mask to ignore padding. Don't truncate the 5% of clips > 11s — truncate them and mark in your ablation study whether this matters.

→ **Total dataset is only 12.4 hours** — this is small. A single training epoch reads all audio in minutes on your RTX 4090. Training bottleneck will be model computation, not data loading.

---

## 5. Valence-Arousal Scatter Plot (Plot 05) — Your Core V-A Space

**What it shows:** Every utterance plotted in 2D V-A space (Valence 1-5, Arousal 1-5), colored by emotion label. This IS the space your soft-label probability matrix will be defined over.

**Key findings — MOST IMPORTANT PLOT FOR YOUR RESEARCH:**

- **V-A correlation is nearly zero (0.005)** — Valence and Arousal are essentially independent dimensions in this dataset. This validates using a 2D V-A grid as your output space (if they were highly correlated, a 1D output would suffice).

- **The center (V=2.5, A=2.5-3.0) is densely populated** — Neutral and No Agreement samples cluster here, making the center region hardest to classify.

- **Data is NOT uniformly distributed** — the upper-left quadrant (High Arousal, Negative Valence) is denser than the lower-right (Low Arousal, Positive Valence). Your probability matrix grid should account for this density imbalance.

- **Discrete clustering is visible:** Emotions don't form clean gaussian blobs — they're smeared across the V-A space with heavy overlap, which is exactly why your soft-label approach (probability distribution over the V-A grid) is better than hard classification.

- **V-A values are quantized** to 0.5 intervals (annotators used a discrete scale) — you see a grid pattern rather than continuous values. This means your probability matrix doesn't need ultra-fine resolution.

**Design decisions:**
→ **Use a 9×9 or 10×10 V-A grid** for your probability matrix (covering 1.0 to 5.0 in 0.5 steps gives 9×9 = 81 bins). Finer resolution (20×20) would have too many empty bins given the quantized annotations.
→ **Gaussian smoothing on ground truth:** Don't assign all probability to a single grid cell. Use a 2D Gaussian kernel (σ≈0.5) centered on the annotated V-A point to create soft targets. This is your "Probabilistic Distribution Embedding."
→ **Loss function for V-A prediction:** Use KL-divergence between predicted and target probability distributions, not MSE on raw V-A values. This aligns with your proposal's soft-label design.

---

## 6. V-A Distribution Per Emotion Class (Plot 06) — Class Separability Analysis

**What it shows:** Each 4-class emotion highlighted against the full V-A scatter, with centroids (black X markers).

**Key findings — CRITICAL FOR UNDERSTANDING MODEL DIFFICULTY:**

- **Angry** — centroid at (V=1.91, A=3.64): Clusters tightly in upper-left quadrant (negative valence, high arousal). **Best separated class** — relatively compact with clear V-A signature.

- **Happy** — centroid at (V=3.95, A=3.41): Clusters in upper-right quadrant (positive valence, high arousal). Well separated from Angry/Sad but **has significant overlap with Neutral** in the mid-valence region.

- **Sad** — centroid at (V=2.25, A=2.56): Lower-left quadrant (negative valence, low arousal). Has a **wide spread** — some sad utterances reach V=4.0, which seems contradictory. This suggests annotation noise or complex sadness expressions (bittersweet, wistful).

- **Neutral** — centroid at (V=2.97, A=2.73): **Sits right in the center** of the V-A space, overlapping with ALL other emotions. This is why Neutral is typically the hardest class to classify correctly — it has no distinctive V-A signature.

**Key insight for your paper:**
The centroids form a meaningful pattern: Angry and Happy have the same arousal (~3.5) but opposite valence (1.9 vs 3.95), while Sad and Neutral have similar arousal (~2.6) but Sad is more negative. This means **Valence is the primary discriminator between positive/negative emotions, while Arousal separates active (Angry, Happy) from passive (Sad, Neutral)**. Your model needs to learn BOTH dimensions well.

**Design decisions:**
→ **Neutral will be your most confused class** — expect 60-70% class-specific accuracy even with a good model. Your paper should analyze the confusion matrix specifically for Neutral.
→ **Angry is your "easy win"** — it has the most distinctive V-A signature. If your model can't get Angry right, something is fundamentally wrong.
→ **The overlap between Sad and Neutral in V-A space** is why your proposal's multi-modal approach matters — text and audio carry disambiguation signals that V-A values alone cannot provide. This is a strong motivation argument for your paper.

---

## 7. Text Length Distribution (Plot 07)

**What it shows:** Word count per transcript, determining your text tokenizer configuration.

**Key findings:**

- **Very short utterances dominate:** Mean = 12 words, Median = 8 words
- **Heavy left skew:** The peak is at 2-5 words (many utterances are single phrases like "Yeah", "Oh come on", "I know")
- **P95 = 32 words** — 95% of utterances have ≤32 words
- **Max = 100 words** — rare long utterances exist but are extreme outliers

**Design decisions:**
→ **Set `max_text_tokens = 48`** (P95 × 1.5 to account for subword tokenization expanding word count). BERT tokenizer typically produces 1.2-1.5 subwords per English word.
→ **This is tiny for BERT** (max 512 tokens) — your text encoder will barely use any of its capacity. This means: (a) text processing will be extremely fast, and (b) text alone carries limited information for short utterances like "Yeah" or "Okay"
→ **The short text length is actually an argument FOR multi-modal fusion** — a 3-word utterance gives BERT almost nothing to work with, but the audio prosody of those same 3 words can be highly emotional. Use this point in your paper.
→ **Consider NOT using BERT's [CLS] token** as the only text representation. For such short sequences, mean-pooling over all token embeddings may capture more signal.

---

## 8. Session Balance (Plot 08) — LOSO Fold Sizes

**What it shows:** Total utterances per session, which determines your cross-validation fold sizes.

**Key findings:**

- **Two groups:** Session 1-2 (~1,815) and Session 3-5 (~2,135). About 15% difference.
- **Mean: 2,008** utterances per session
- This imbalance is mild and acceptable for LOSO — no special handling needed

**Design decisions:**
→ **Each LOSO fold:** Train on ~8,000 samples, test on ~2,000. With batch size 32, that's ~250 training steps per epoch — very fast on RTX 4090.
→ **Each session = 1 unique actor pair (1M + 1F).** LOSO ensures you test on speakers never seen during training. This is why IEMOCAP's LOSO protocol is considered speaker-independent evaluation, which is much harder (and more realistic) than random splits.
→ **Never randomly split IEMOCAP** — this would leak speaker identity information across train/test and inflate your results by 5-10%. Reviewers will immediately reject papers that don't use LOSO.

---

## 9. Modality Completeness (Plot 09)

**What it shows:** Whether all samples have the required input modalities.

**Key finding:**
→ **100% completeness** — all 10,039 utterances have both audio (.wav) and text (transcript). This is ideal. No missing data handling needed. Your dataloader can assume every sample has both modalities.

---

## Summary: Concrete Configuration for Your Model

Based on all the analysis above, here are the recommended hyperparameters:

| Parameter | Value | Reason |
|-----------|-------|--------|
| Classification setting | 4-class (Angry, Happy+Excited, Sad, Neutral) | Standard benchmark |
| Evaluation protocol | 5-fold LOSO | Required for IEMOCAP |
| Max audio length | 11 seconds (176,000 samples at 16kHz) | P95 coverage |
| Audio feature frames | ~550 (after Wav2Vec2.0 downsampling) | Mamba input length |
| Max text tokens | 48 | P95 × 1.5 subword expansion |
| V-A grid resolution | 9×9 (0.5 step from 1.0 to 5.0) | Matches annotation quantization |
| V-A soft target σ | 0.5 | Gaussian smoothing for probability matrix |
| Classification loss | CrossEntropy with class weights [1.0, 0.67, 1.0, 1.02] | Mild imbalance correction |
| V-A loss | KL-divergence | Soft-label distribution matching |
| Batch size | 32 | Fits RTX 4090 with frozen encoders |
| Training steps/epoch | ~250 (for 4-class, ~5,500 train samples per fold) | Fast iteration |
| Usable samples (4-class) | 5,531 | After filtering |
| Usable samples (6-class) | 7,380 | Alternative setting |

---

## Next Steps

1. **Build the PyTorch Dataset/DataLoader** using the CSV export (`iemocap_utterances.csv`) as the index
2. **Implement the LOSO splitter** — a function that yields 5 train/test splits by session
3. **Start with BERT + Wav2Vec2.0 late fusion baseline** — get a working training loop first
4. **Then swap in Mamba blocks** and compare efficiency + accuracy
