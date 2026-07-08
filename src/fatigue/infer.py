"""
Stage 7 — inference. This is the function the scheduler and the agent call.

The saved model is a full sklearn Pipeline (StandardScaler + classifier), so no
separate scaler object is needed at inference time.
"""
from .features import extract_features


def predict_fatigue(wav_path, speaker_id, model, baselines):
    """
    Given a new voice check-in, return the worker's fatigue level.
      1. Extract 80 features
      2. Subtract the speaker's enrollment baseline  -> delta vector
      3. Run through the pipeline                     -> class probabilities
      4. Collapse to a continuous fatigue score in [0, 1]
    """
    feats = extract_features(wav_path)
    if feats is None:
        raise ValueError("Audio too short or unreadable.")
    if speaker_id not in baselines:
        raise ValueError(
            f"No baseline for speaker '{speaker_id}'. "
            "Enroll them first with at least one rested clip.")

    delta = (feats - baselines[speaker_id]).reshape(1, -1)
    proba = model.predict_proba(delta)[0]        # [P_alert, P_mild, P_fatigued]

    # Continuous score in [0, 1], weighted by severity.
    fatigue_score = float(proba[1] * 0.5 + proba[2] * 1.0)

    if fatigue_score < 0.35:
        status = "alert"
    elif fatigue_score < 0.70:
        status = "mild_fatigue"
    else:
        status = "fatigued"

    return {
        "fatigue_score": round(fatigue_score, 3),
        "status": status,
        "probabilities": {
            "alert":        round(float(proba[0]), 3),
            "mild_fatigue": round(float(proba[1]), 3),
            "fatigued":     round(float(proba[2]), 3),
        },
    }
