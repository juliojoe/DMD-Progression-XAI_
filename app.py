import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="DMD Progression Predictor (XAI)", page_icon="🧬", layout="wide")


@st.cache_resource
def load_artifacts(path="dmd_model.joblib"):
    return joblib.load(path)


@st.cache_resource
def get_explainer(_clf, _background):
    if hasattr(_clf, "coef_"):
        return shap.LinearExplainer(_clf, shap.maskers.Independent(_background, max_samples=len(_background)))
    return shap.TreeExplainer(_clf)


try:
    art = load_artifacts()
except FileNotFoundError:
    st.error("`dmd_model.joblib` not found. Run the whole notebook first (section 10 exports it) "
             "and keep it in the same folder as this app.")
    st.stop()

pipe = art["pipe"]
pre = pipe.named_steps["pre"]
clf = pipe.named_steps["clf"]
class_names, POS = art["class_names"], art["pos_index"]
num_cols, cat_cols = art["num_cols"], art["cat_cols"]
strength_cols, rom_cols = art["strength_cols"], art["rom_cols"]
ordinal, pretty = art["ordinal_labels"], art["pretty"]
feat_names = art["feat_names"]
derived = ["Strength_composite_z", "ROM_composite_z"]


def display_matrix(Xdf):
    # imputed (not scaled) numeric values + one-hot categories, so plots show real units
    num_part = pre.named_transformers_["num"].named_steps["imp"].transform(Xdf[num_cols])
    cat_part = pre.named_transformers_["cat"].transform(Xdf[cat_cols])
    return np.hstack([num_part, cat_part])


st.title("🧬 " + art["title"])
st.caption(f"Model: {art['model_name']}  |  Data: open KU Leuven DMD dataset (doi 10.48804/ONRNG0)  |  "
           "Research prototype, not a clinical decision tool.")

tab_pred, tab_perf, tab_about = st.tabs(["Predict & explain", "Model performance", "About"])

# ---------------------------------------------------------------- Predict
with tab_pred:
    st.markdown(f"Estimates the chance that this boy's **composite lower-limb strength z-score falls by at least "
                f"{art['decline_threshold']} by his next visit**.")
    threshold = st.slider("Decision threshold for 'Strength decline'", 0.05, 0.95, 0.50, 0.05)

    values = {}
    stats = art["num_stats"]

    def num_input(container, c, step=0.1, fmt="%.2f"):
        s = stats[c]
        lo, hi = float(s["min"]), float(s["max"])
        if hi <= lo:
            hi = lo + 1.0
        return container.number_input(pretty.get(c, c), min_value=lo, max_value=hi,
                                      value=float(s["median"]), step=step, format=fmt, key=f"n_{c}")

    with st.form("patient_form"):
        st.markdown("**Muscle strength z-scores** (0 = typically developing peers; negative = weaker)")
        cols = st.columns(4)
        for i, c in enumerate(strength_cols):
            values[c] = num_input(cols[i % 4], c)

        st.markdown("**Range of motion z-scores**")
        cols = st.columns(3)
        for i, c in enumerate(rom_cols):
            values[c] = num_input(cols[i % 3], c)

        st.markdown("**Clinical profile**")
        cols = st.columns(4)
        for i, c in enumerate(["Age_category", "BodyMass_category", "Height_category", "BMI_category"]):
            labels = ordinal[c]
            codes = sorted(labels)
            default = min(codes, key=lambda k: abs(k - stats[c]["median"]))
            choice = cols[i].selectbox(pretty[c], [labels[k] for k in codes], index=codes.index(default), key=f"o_{c}")
            values[c] = [k for k in codes if labels[k] == choice][0]

        cols = st.columns(4)
        values["Mutation Type"] = cols[0].selectbox("Mutation type", art["cat_options"]["Mutation Type"])
        values["Corticosteroids"] = cols[1].selectbox("Corticosteroid", art["cat_options"]["Corticosteroids"])
        values["Session"] = num_input(cols[2], "Session", step=1.0, fmt="%.0f")
        values["prev_change"] = num_input(cols[3], "prev_change")

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        values["Strength_composite_z"] = float(np.mean([values[c] for c in strength_cols]))
        values["ROM_composite_z"] = float(np.mean([values[c] for c in rom_cols]))
        row = pd.DataFrame([values], columns=num_cols + cat_cols)
        for c in cat_cols:
            row[c] = row[c].astype(object)

        p_dec = float(pipe.predict_proba(row)[0, POS])
        left, right = st.columns(2)
        with left:
            st.metric("Probability of strength decline", f"{p_dec:.1%}")
            if p_dec >= threshold:
                st.error(f"Predicted: **{class_names[POS]}**  (threshold {threshold:.0%})")
            else:
                st.success(f"Predicted: **{class_names[1 - POS]}**  (threshold {threshold:.0%})")
            st.progress(min(max(p_dec, 0.0), 1.0))
            st.caption(f"Composite strength z entered: {values['Strength_composite_z']:.2f}. "
                       "The model was trained on very few boys, so treat this as an illustration of explainable "
                       "prediction, not as a forecast.")

        with right:
            st.markdown("**Why this risk? (SHAP)**")
            t = pre.transform(row)
            explainer = get_explainer(clf, art["background"])
            sv = explainer.shap_values(t) if hasattr(clf, "coef_") else explainer.shap_values(t, check_additivity=False)
            sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
            base = np.atleast_1d(explainer.expected_value)
            if sv.ndim == 2:
                sv = np.stack([-sv, sv], axis=-1)
                base = np.array([-base[0], base[0]])

            plt.figure()
            shap.plots.waterfall(
                shap.Explanation(values=sv[0, :, POS], base_values=base[POS],
                                 data=display_matrix(row)[0], feature_names=feat_names),
                max_display=12, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            st.caption("Red bars push toward strength decline; blue bars push toward stable or improving strength.")

# ------------------------------------------------------------ Performance
with tab_perf:
    st.subheader("Patient-level cross-validation results")
    st.info(art["cv_note"])
    res = pd.DataFrame(art["results"]).set_index("Model")
    st.dataframe(res.style.format("{:.3f}"), width="stretch")

    st.subheader("Global importance of clinical variables (SHAP)")
    gi = pd.Series(art["global_importance"]).sort_values()
    gi.index = [pretty.get(i, i) for i in gi.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    gi.plot.barh(ax=ax, color="#2a7f62")
    ax.set_xlabel("summed mean |SHAP value|")
    st.pyplot(fig, clear_figure=True)

# ------------------------------------------------------------------ About
with tab_about:
    st.markdown(
        """
**Project:** Explainable AI-Based Duchenne Muscular Dystrophy Progression Prediction Using Machine Learning

**Data:** open KU Leuven Research Data Repository dataset (https://doi.org/10.48804/ONRNG0) from Vandekerckhove et al.,
*Longitudinal trajectories of muscle impairments in growing boys with Duchenne muscular dystrophy* (medRxiv 2024).
Please cite the dataset and the paper.

**Method:** consecutive visits of the same boy form visit pairs; the label is a drop of at least 0.25 z in composite
lower-limb strength at the next visit. Models are evaluated with repeated cross-validation **split by boy**, and
explained with SHAP and LIME.

**Limitations:** 33 boys, about 100 visit pairs, unknown time between visits, coarse age bands, noisy strength
measurements. SHAP shows what the model relies on, not biological causation. Not for clinical use.
"""
    )
