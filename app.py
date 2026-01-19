import streamlit as st
import pandas as pd

from common import *

class_dict = {
    0: "Background",
    1: "Left Hippocampus",
    2: "Left External Capsule",
    3: "Left Caudate Putamen",
    4: "Left Anterior Commissure",
    5: "Left Globus Pallidus",
    6: "Left Internal Capsule",
    7: "Left Thalamus",
    8: "Left Cerebellum",
    9: "Left Superior Colliculi",
    10: "Left Ventricles",
    11: "Left Hypothalamus",
    12: "Left Inferior Colliculi",
    13: "Left Central Gray",
    14: "Left Neocortex",
    15: "Left Amygdala",
    16: "Left Olfactory bulb",
    17: "Left Brain Stem",
    18: "Left Rest of Midbrain",
    19: "Left Basal Forebrain Septum",
    20: "Left Fimbria",
    21: "Right Hippocampus",
    22: "Right External Capsule",
    23: "Right Caudate Putamen",
    24: "Right Anterior Commissure",
    25: "Right Globus Pallidus",
    26: "Right Internal Capsule",
    27: "Right Thalamus",
    28: "Right Cerebellum",
    29: "Right Superior Colliculi",
    30: "Right Ventricles",
    31: "Right Hypothalamus",
    32: "Right Inferior Colliculi",
    33: "Right Central Gray",
    34: "Right Neocortex",
    35: "Right Amygdala",
    36: "Right Olfactory bulb",
    37: "Right Brain Stem",
    38: "Right Rest of Midbrain",
    39: "Right Basal Forebrain Septum",
    40: "Right Fimbria",
}

mapping_dict = {
    0: 0,
    1: 1,
    2: 15,
    3: 3,
    4: 15,
    5: 3,
    6: 15,
    7: 10,
    8: 16,
    9: 9,
    10: 17,
    11: 10,
    12: 9,
    13: 9,
    14: 12,
    15: 5,
    16: 7,
    17: 9,
    18: 9,
    19: 12,
    20: 15,
    21: 2,
    22: 14,
    23: 4,
    24: 14,
    25: 4,
    26: 14,
    27: 11,
    28: 16,
    29: 9,
    30: 17,
    31: 11,
    32: 9,
    33: 9,
    34: 13,
    35: 6,
    36: 8,
    37: 9,
    38: 9,
    39: 13,
    40: 14,
}

mouse_mapping = torch.zeros(max(mapping_dict.keys()) + 1, dtype=torch.long)
for k, v in mapping_dict.items():
    mouse_mapping[k] = v

unified_class_dict = {
    0: "Background",
    1: "Left Hippocampus",
    2: "Right Hippocampus",
    3: "Left Basal Ganglia",
    4: "Right Basal Ganglia",
    5: "Left Amygdala",
    6: "Right Amygdala",
    7: "Left Olfactory bulb",
    8: "Right Olfactory bulb",
    9: "Brain stem and Midbrain",
    10: "Left Thalamus and Hypothalamus",
    11: "Right Thalamus and Hypothalamus",
    12: "Left Neocortex",
    13: "Right Neocortex",
    14: "Right Forebrain White Matter",
    15: "Left Forebrain White Matter",
    16: "Cerebellum",
    17: "Ventricles",
}

@st.cache_data
def load_metrics():
    df = pd.read_csv('app_data/metrics.csv', index_col=0)#.iloc[1:]
    return df

@st.cache_data
def load_cm():
    return torch.load("app_data/cm.pt")

@st.cache_data(max_entries=5)
def load_viz(i):
    data = torch.load(f"app_data/volume_{i}.pt", map_location="cpu", weights_only=False)
    return [data["template"]], [data["pred"]], [data["label"]]

N_CLASSES = len(class_dict)

st.set_page_config(layout="wide")
st.text('Authors: Matej Polak, Miroslav Cepek (FIT CTU). Jan 2026.')
st.title('Mouse brain segmentation: Results')
st.markdown("""
We have trained a UNet model for automatic segmentation of mouse brain MRIs into 40 regions.
We used a dataset of 10 labeled mouse MRI scans (in vivo, isotropic 100 μm, T2, details [here](10.3389/neuro.05.001.2008)).
The results are presented in this interactive report.'
In the future, we wish to employ this trained MOUSE model onto segmentation of RABBIT scans.
""")

# Visualization
col_left, col_right = st.columns([4, 1])
with col_left:
    st.subheader("Visualization")
with col_right:
    show_unified = st.checkbox("Show unified", value=False)
st.markdown("""
This interactive visualization shows the brain sliced in the coronal direction, in which the scans were taken.
- **Volume index**: switches between the brains.
- **Y**: moves the visualizations along the coronal axis.
- **Class**: Enables to filter for a specific class only.
- **Template**: Original MRI scan.
- **Ground truth**: Manually labeled regions by the creators of the dataset.
- **Prediction**: Prediction of our model.
- **Overlap**: Red shows where the ground truth and the prediction disagree. This is usually happening on borders between areas.
""")

viz_streamlit(
    pred=None,
    gt=None,
    template=None,
    show_template=True,
    class_dict=unified_class_dict if show_unified else class_dict,
    load_viz=load_viz,
    show_unified=show_unified,
    mapping_dict=mapping_dict,
)

# Metrics
st.subheader("Metrics")
st.write("The metrics to follow are IOU and F1. "
         "Performance on underrepresented regions "
         "(such as Anterior Commissure or Internal Capsule) "
         "drops significantly.")

df_metrics = load_metrics()
metric_cols = ["IOU (Jaccard)", "F1 (Dice)", "Accuracy"]

if show_unified:
    df_metrics = map_df(df_metrics, mapping_dict, unified_class_dict, metric_cols)

sum_row = {"Class name": "Total", "Pixel count": df_metrics["Pixel count"].iloc[1:].sum()}

for c in metric_cols:
    mask = (df_metrics["Pixel count"] > 0) & (df_metrics.index > 0)
    sum_row[c] = np.average(df_metrics[c][mask].to_numpy(), weights=df_metrics["Pixel count"][mask].to_numpy())
df_total = pd.DataFrame(sum_row, index=[0])

for col in metric_cols:
    df_metrics[col] = (
            df_metrics[col] * 100
    ).map(lambda x: f"{x:.2f} %")
    df_total[col] = (
            df_total[col] * 100
    ).map(lambda x: f"{x:.2f} %")
st.dataframe(df_metrics.iloc[1:], width='stretch', hide_index=False)
st.dataframe(df_total, width='stretch', hide_index=False)

pixel_map = (
    df_metrics
    .assign(
        key=lambda df: df.index.astype(str) + ": " + df["Class name"]
    )
    .set_index("key")["Pixel count"]
)
pixel_counts = df_metrics["Pixel count"].to_dict()

# Confusion matrix
col_left, col_right = st.columns([4, 1])
with col_left:
    st.subheader("Confusion matrix")
with col_right:
    normalize = st.checkbox("Normalize", value=True)
st.markdown("""
Confusion matrix shows the actual labels (Y-axis) and the predict labels (X-axis).
In brackets is the total number of pixels in datasets for given region.
Ideally, everything should be on the diagonal (correct prediction, true label = predicted label).
Blue spots outside the diagonal mark an error (confusion) in model predictions. Note that the errors mostly occur on underrepresented regions.
For better inspection, the major 20 confusions are listed in the table below.
""")
cm = load_cm()

if show_unified:
    cm = map_cm(cm, mapping_dict)
if normalize:
    cm = normalize_cm(cm)
plot_cm(cm, class_dict=unified_class_dict if show_unified else class_dict, pixel_counts=pixel_counts, streamlit=True)

# Top confusions
st.subheader("Top confusions")
top_confusions = get_top_confusions(cm, class_dict=unified_class_dict if show_unified else class_dict)
df_conf = pd.DataFrame(top_confusions)
df_conf = df_conf.rename(columns={
    "true_class": "True class",
    "predicted_as": "Predicted as",
    "value": "Confusion"
})

df_conf["Pixel count"] = df_conf["True class"].map(pixel_map)
if normalize:
    df_conf["Confusion"] = (
        (df_conf["Confusion"] * 100).map(lambda x: f"{x:.2f} %")
        + " (out of "
        + df_conf["Pixel count"].astype(int).map("{:,}".format)
        + ")"
    )
else:
    df_conf["Confusion"] = (
        df_conf["Confusion"].astype(int).map("{:,}".format)
        + " (out of "
        + df_conf["Pixel count"].astype(int).map("{:,}".format)
        + ")"
    )
df_conf = df_conf.drop(columns=["Pixel count"])
st.dataframe(df_conf, width='stretch', hide_index=True)

st.subheader("Questions")
st.markdown("""
- Do the observed confusions make sense? Are the regions anatomically similar?
- Are the observed misclassifications to be expected in human labeling as well?
- Are any of the present regions easier/harder for recognition? Does this align with the obtained results?
- How important are the underrepresented regions for your own use cases?
- What are the biggest anatomical differences between mouse brain and rat brain?
""")