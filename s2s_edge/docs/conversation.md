# Multilingual conversations, proper names and mixed-language speech

## 1. Turn-level language switching (works): `--pair vi,ko` and `--auto`

Every utterance is routed on its own, so speakers can switch language between turns, and the partner language can change during a session.

```bash
python -m s2s_edge live --pair vi,ko  --models models/models.json     # Vietnamese <-> Korean, direction chosen per utterance
python -m s2s_edge live --auto --partner en --models models/models.json   # Vietnamese <-> whichever of en / zh / ko is spoken
python -m s2s_edge file --auto --in meeting.wav --out meeting_translated.wav   # a recording with several languages
```

Language identification costs no extra model:

1. The Vietnamese CTC model (Zipformer, the cheapest ASR, ~174 ms on chip) runs first. A logistic regression over six confidence features
   (`assets/lid_vi.json`: mean max-probability on non-blank frames, share of decoded words that are valid Vietnamese syllables, non-blank ratio,
   mean max-probability, mean blank probability, word count) says "Vietnamese or not".
2. Not Vietnamese + one candidate language → that language. Several candidates (`--auto`) → SenseVoice runs in `auto` mode and the script of its text
   decides: Hangul → ko, Han → zh, Latin → en. Its text is reused as the transcript, so there is no second ASR pass.
3. In `--auto` mode the partner language is sticky: Vietnamese speech is translated into the language last heard from the other side (`--partner` sets the first one).

Measured on FLEURS test read speech (30 clips per language, `tools/lid_features.py`, CPU reference):

| | result |
|---|---|
| Vietnamese vs not (clean, 8–15 s clips) | 10-fold cross-validation **120 / 120** correct; vi clips probability ≥ 0.70, other clips ≤ 0.31 |
| en / zh / ko from the script of SenseVoice(auto) text | **90 / 90** correct |
| Margin of a single threshold (`conf_nb`) | thin: vi ≥ 0.765, others ≤ 0.753, which is why the six-feature model is used |

Held-out clips (10 per language, not used to fit the model, `tools/lid_robust.py`):

| Condition | vi recognised | en / zh / ko never called Vietnamese |
|---|---|---|
| full clip | 10 / 10 | yes (max probability 0.12) |
| 3 s crop | 10 / 10 | yes (0.26) |
| 1.5 s crop | 6 / 10 | yes (0.10) |
| 3 s + babble at 5 dB | 7 / 10 | yes (0.08) |

Errors are one-sided: very short or noisy Vietnamese can be missed (a positive `bias` does not recover them), foreign speech is never taken for Vietnamese.
The segmenter therefore merges pauses shorter than 1.5 s around short segments, and fragments without intelligible text are dropped instead of translated.

Integration test, one recording with six turns (vi, ko, vi, en, vi, zh) in `--auto` mode (`samples/mixed_conversation.wav`): all six turns were routed correctly
(vi→en with the default partner, ko→vi, vi→ko after the Korean turn, en→vi, vi→en after the English turn, zh→vi).

Not yet measured: real conversational utterances (this test uses read speech), and Vietnamese loanwords inside foreign speech.
`tools/lid_features.py` collects the features; `LanguageID(bias=...)` shifts the decision if the field test shows a systematic error.

## 2. Proper names

What breaks, per stage:

| Stage | Problem | What we do |
|---|---|---|
| ASR | names are misspelled (no language model for names) | nothing general; `_asr_fix` entries in the glossary repair known terms (`"ếch xen" → "Excel"`) |
| MT (NLLB) | usually copies or transliterates correctly (`Recep Tayyip Erdogan → 雷杰普·塔伊普·埃尔多安`, `Samsung → 三星`) | kept as is |
| TTS vi (Piper) | **silently drops every word outside its syllable table** (names, foreign words) | `names.py`: such Latin words are respelled as Vietnamese syllables (CMUdict phonemes, letter rules for other origins), the way Vietnamese speakers read them: `Pamela Ferguson → pe mơ lơ phơ gơ sơn`. Words Piper already knows are left alone. 84 syllables of 31 test names: 6 outside Piper's table |
| TTS zh (Melo) | drops Latin letters | leftover Latin names are transliterated by NLLB itself (name alone, English → zh), cached |
| TTS ko (Supertonic) | Latin names | same, English → ko |
| TTS en (Supertonic) | none | Latin names pass through |

A JSON glossary (`--glossary glossary.example.json`) overrides all of this per name: `{"Samsung": {"vi": "sam sung", "zh": "三星", "ko": "삼성"}}`.

## 3. Mixed languages inside ONE utterance (limits)

* zh + en in one sentence: SenseVoice handles it natively.
* Vietnamese with English words ("check giúp em file Excel"): Vietnamese CTC spells English words as Vietnamese-sounding syllables, and the translator then
  sees nonsense. Mitigation available today: glossary `_asr_fix` for the terms of the deployment (factory vocabulary). A real fix needs a bilingual ASR or
  recordings of code-switched factory speech to fine-tune/measure on; none were available, so this is **not solved and not measured**.
* Several speakers talking at once: not handled (single stream, no diarisation).
