"""The house-price app on Hugging Face: it loads the version of the model
that the pipeline published, and answers one house at a time."""
import gradio as gr
import joblib
import pandas as pd
import spaces
from huggingface_hub import hf_hub_download

# Both files are written by the pipeline before it uploads this folder.
with open("model_repo.txt") as f:
    MODEL_REPO = f.read().strip()
with open("model_version.txt") as f:
    VERSION = f.read().strip()

model = joblib.load(hf_hub_download(MODEL_REPO, "model.joblib", revision=VERSION))
FEATURES = ["size_m2", "rooms", "age_years"]


# ZeroGPU Spaces must have at least one function marked @spaces.GPU.
# This model is small and does not need the GPU; the mark is only the rule.
@spaces.GPU
def predict_price(size, rooms, age):
    house = pd.DataFrame([[size, rooms, age]], columns=FEATURES)
    return f"€{model.predict(house)[0] * 1000:,.0f}"


demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(30, 300, value=100, step=5, label="Size (m²)"),
        gr.Slider(1, 6, value=3, step=1, label="Rooms"),
        gr.Slider(0, 60, value=15, step=1, label="Age (years)"),
    ],
    outputs=gr.Textbox(label="Predicted price"),
    examples=[[60, 2, 30], [100, 3, 15], [150, 4, 5]],
    title="House Price Prediction",
    description=f"Model {MODEL_REPO}, version {VERSION}, tested and published by a GitHub pipeline. For learning only.",
    flagging_mode="never",
)
demo.launch()
