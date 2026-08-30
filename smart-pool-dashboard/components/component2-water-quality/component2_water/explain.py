"""
Component 2 — human-readable explanation for a SAFE/WARNING/CRITICAL result.

The XGBoost classifier in predictor.py only returns a class label; it has
no built-in feature attribution. These thresholds are standard swimming-pool
water-quality guidance, not something the model outputs. Turbidity's cut
points (1.0 / 5.0 NTU) also match what the training dataset itself shows --
ml/datasets/pool_water_quality_augmented_dataset.csv has CRITICAL rows
almost entirely defined by turbidity > 5 NTU, with pH/chlorine/TDS staying
inside a normal band even in CRITICAL rows. Turbidity is the dominant real
signal in this dataset; the other bounds below are domain knowledge filled
in for parameters the training data didn't vary much.
"""

PH_IDEAL = (7.2, 7.8)
PH_ACCEPTABLE = (6.8, 8.0)

CHLORINE_IDEAL = (1.0, 3.0)
CHLORINE_ACCEPTABLE = (0.5, 5.0)

TURBIDITY_IDEAL_MAX = 1.0
TURBIDITY_CRITICAL_MIN = 5.0

TDS_IDEAL_MAX = 1000.0
TDS_CRITICAL_MIN = 1500.0


def explain_status(reading: dict, status: str) -> list:
    """reading keys: ph, temperature, chlorine, turbidity, tds.

    Returns human-readable reasons for a WARNING/CRITICAL classification,
    most-out-of-range first. Empty list for SAFE/UNKNOWN.
    """
    if status in ("SAFE", "UNKNOWN"):
        return []

    reasons = []
    ph = reading.get("ph")
    chlorine = reading.get("chlorine")
    turbidity = reading.get("turbidity")
    tds = reading.get("tds")

    if turbidity is not None:
        if turbidity >= TURBIDITY_CRITICAL_MIN:
            reasons.append(
                f"Turbidity is very high ({turbidity:.2f} NTU) — water is visibly "
                f"cloudy, above the {TURBIDITY_CRITICAL_MIN:.0f} NTU safety limit"
            )
        elif turbidity > TURBIDITY_IDEAL_MAX:
            reasons.append(
                f"Turbidity is elevated ({turbidity:.2f} NTU) — above the "
                f"{TURBIDITY_IDEAL_MAX:.0f} NTU clear-water target"
            )

    if ph is not None:
        if ph < PH_ACCEPTABLE[0]:
            reasons.append(
                f"pH is very low ({ph:.2f}) — water is too acidic, below the "
                f"safe minimum of {PH_ACCEPTABLE[0]}"
            )
        elif ph > PH_ACCEPTABLE[1]:
            reasons.append(
                f"pH is very high ({ph:.2f}) — water is too alkaline, above the "
                f"safe maximum of {PH_ACCEPTABLE[1]}"
            )
        elif ph < PH_IDEAL[0] or ph > PH_IDEAL[1]:
            reasons.append(
                f"pH is outside the ideal range ({ph:.2f}, target "
                f"{PH_IDEAL[0]}-{PH_IDEAL[1]})"
            )

    if chlorine is not None:
        if chlorine < CHLORINE_ACCEPTABLE[0]:
            reasons.append(
                f"Chlorine is very low ({chlorine:.2f} ppm) — insufficient "
                f"disinfection, below {CHLORINE_ACCEPTABLE[0]} ppm"
            )
        elif chlorine > CHLORINE_ACCEPTABLE[1]:
            reasons.append(
                f"Chlorine is very high ({chlorine:.2f} ppm) — above the safe "
                f"maximum of {CHLORINE_ACCEPTABLE[1]} ppm"
            )
        elif chlorine < CHLORINE_IDEAL[0] or chlorine > CHLORINE_IDEAL[1]:
            reasons.append(
                f"Chlorine is outside the ideal range ({chlorine:.2f} ppm, "
                f"target {CHLORINE_IDEAL[0]}-{CHLORINE_IDEAL[1]})"
            )

    if tds is not None:
        if tds >= TDS_CRITICAL_MIN:
            reasons.append(
                f"TDS is very high ({tds:.0f} ppm) — above the "
                f"{TDS_CRITICAL_MIN:.0f} ppm safety limit"
            )
        elif tds > TDS_IDEAL_MAX:
            reasons.append(
                f"TDS is elevated ({tds:.0f} ppm) — above the "
                f"{TDS_IDEAL_MAX:.0f} ppm target"
            )

    if not reasons:
        reasons.append(
            f"Classifier flagged {status} from the combined sensor readings, "
            "though no single parameter is far outside its normal band"
        )

    return reasons
