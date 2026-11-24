"""EcoSort — Smart Waste Classification & Recycling Assistant.

The pretrained ten-class image classifier identifies the likely material/object.
A transparent rule-based mapping then groups it into one of three sorting groups:
Recyclable, Organic, or Residual. Battery predictions receive a special-disposal
warning so they are never presented as ordinary household residual waste.
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, SiglipForImageClassification


MODEL_ID = "prithivMLmods/Augmented-Waste-Classifier-SigLIP2"

# The model's categories are read from its own configuration during prediction.
# These keys are the category names expected by the current pretrained model.
WASTE_INFO: dict[str, dict[str, str]] = {
    "battery": {
        "title": "Battery",
        "icon": "🔋",
        "colour": "#D69B36",
    },
    "biological": {
        "title": "Biological / organic material",
        "icon": "🍃",
        "colour": "#6EAA62",
    },
    "cardboard": {
        "title": "Cardboard",
        "icon": "📦",
        "colour": "#D7A45B",
    },
    "clothes": {
        "title": "Clothes / textiles",
        "icon": "👕",
        "colour": "#9C83C7",
    },
    "glass": {
        "title": "Glass",
        "icon": "🍾",
        "colour": "#61A889",
    },
    "metal": {
        "title": "Metal",
        "icon": "🥫",
        "colour": "#92A4B7",
    },
    "paper": {
        "title": "Paper",
        "icon": "📄",
        "colour": "#6C9BD2",
    },
    "plastic": {
        "title": "Plastic",
        "icon": "🧴",
        "colour": "#E4B84F",
    },
    "shoes": {
        "title": "Shoes",
        "icon": "👟",
        "colour": "#B18A6B",
    },
    "trash": {
        "title": "General / mixed trash",
        "icon": "🗑️",
        "colour": "#8B919B",
    },
}

# The three main sorting groups selected for EcoSort.
SORTING_GROUPS: dict[str, dict[str, str]] = {
    "Recyclable": {
        "icon": "♻️",
        "colour": "#23845A",
        "description": "Materials that may be recycled or sent to a reuse/recovery stream when clean and accepted locally.",
    },
    "Organic": {
        "icon": "🍃",
        "colour": "#6E9F42",
        "description": "Biodegradable food or plant waste that may be composted or collected separately where facilities exist.",
    },
    "Residual": {
        "icon": "🗑️",
        "colour": "#8B919B",
        "description": "Waste that is not suitable for ordinary recycling or composting. Check local rules before disposal.",
    },
}

# Material-to-group mapping. This is a simple, explainable rule-based step;
# it does not change or retrain the pretrained image classifier.
CATEGORY_ROUTING: dict[str, dict[str, str]] = {
    "battery": {
        "group": "Residual",
        "stream": "Special battery collection — not the ordinary residual bin",
        "guidance": "Keep batteries separate. Do not crush, puncture or burn them. Take them to an authorised battery collection or take-back point. This special instruction overrides the general three-way sorting label.",
        "warning": "Special handling required: never place batteries in ordinary household residual waste or recycling bins.",
    },
    "biological": {
        "group": "Organic",
        "stream": "Organic waste / composting (where accepted)",
        "guidance": "Keep food scraps and plant waste separate from dry recyclables. Compost them or use the organic-waste collection service where available.",
        "warning": "Avoid mixing organic waste with clean paper, plastic, glass or metal recyclables.",
    },
    "cardboard": {
        "group": "Recyclable",
        "stream": "Paper and cardboard recycling (where accepted)",
        "guidance": "Keep cardboard clean and dry. Flatten boxes and remove tape or non-paper parts when practical. Follow local collection rules.",
        "warning": "Wet, oily or heavily food-soiled cardboard may not be accepted in normal paper recycling.",
    },
    "clothes": {
        "group": "Recyclable",
        "stream": "Textile reuse, donation or textile collection",
        "guidance": "Reuse, repair or donate clean wearable clothes first. For damaged textiles, use a textile collection or recovery programme where one is available; do not assume they belong in ordinary paper/plastic recycling.",
        "warning": "This is a reuse/textile-recovery recommendation, not a guarantee that local curbside recycling accepts clothes.",
    },
    "glass": {
        "group": "Recyclable",
        "stream": "Glass collection / recycling (where accepted)",
        "guidance": "Keep glass separate and follow local glass-collection rules. Handle broken glass carefully and package it safely according to local guidance.",
        "warning": "Not all types of glass are accepted in the same recycling stream.",
    },
    "metal": {
        "group": "Recyclable",
        "stream": "Metal recycling (where accepted)",
        "guidance": "Empty containers and rinse them when practical. Keep sharp edges safe and check whether your local recycler accepts the item.",
        "warning": "Pressurised or hazardous containers may need a separate disposal route.",
    },
    "paper": {
        "group": "Recyclable",
        "stream": "Paper recycling (where accepted)",
        "guidance": "Keep paper clean and dry. Paper soaked with oil, grease or food may not be accepted with ordinary paper recycling.",
        "warning": "Coated, laminated or heavily soiled paper may need a different disposal route.",
    },
    "plastic": {
        "group": "Recyclable",
        "stream": "Plastic recycling (where accepted)",
        "guidance": "Empty and rinse the item. Check the plastic type, recycling symbol and local collection rules; not every plastic item is accepted everywhere.",
        "warning": "A plastic identification does not automatically mean that the item is recyclable in your area.",
    },
    "shoes": {
        "group": "Recyclable",
        "stream": "Footwear reuse, donation or textile collection",
        "guidance": "Reuse, repair or donate wearable shoes first. For worn-out footwear, check for a local footwear or textile take-back scheme.",
        "warning": "This is a reuse/recovery recommendation; ordinary recycling services may not accept shoes.",
    },
    "trash": {
        "group": "Residual",
        "stream": "Residual / general waste (verify locally)",
        "guidance": "Use the residual-waste stream only when the item cannot be reused, recycled or composted. Check whether it needs a special collection route before disposal.",
        "warning": "The model's trash class is broad. It does not prove that an item is non-recyclable or safe for ordinary waste.",
    },
}


# Conservative review thresholds for whole-image predictions.
# These are practical guard rails for this prototype, not validated accuracy metrics.
MIN_TOP_SCORE = 0.30          # 30% softmax score
MIN_TOP_TWO_MARGIN = 0.08     # 8 percentage-point gap from the next class


def assess_prediction(predictions: list[dict[str, Any]]) -> tuple[bool, str, float]:
    """Flag low-confidence or ambiguous whole-image predictions for manual review."""
    if not predictions:
        return True, "The model did not return any material scores.", 0.0

    top_score = float(predictions[0]["confidence"])
    second_score = float(predictions[1]["confidence"]) if len(predictions) > 1 else 0.0
    margin = top_score - second_score

    if top_score < MIN_TOP_SCORE:
        return (
            True,
            "The top model score is low. The image may contain mixed waste, several objects, "
            "or an item the model does not recognise clearly.",
            margin,
        )

    if margin < MIN_TOP_TWO_MARGIN:
        return (
            True,
            "The leading material categories have very similar scores, so the model cannot "
            "separate them confidently.",
            margin,
        )

    return False, "", margin


st.set_page_config(
    page_title="EcoSort | Smart Waste Assistant",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Nunito:wght@500;600;700;800;900&display=swap');
      :root {
        --ink:#244438; --forest:#174B3A; --green:#247C57; --green-2:#32976D;
        --lime:#C9EC79; --mint:#EAF5E8; --cream:#FBF8EF; --paper:#FFFFFF;
        --muted:#718078; --line:#E5EBDD; --peach:#FBE2D0; --sky:#E0EEF5;
        --lavender:#EAE6F6; --sand:#F5E9D4;
      }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      .stApp {
        background:
          radial-gradient(circle at 8% 4%, rgba(201,236,121,.18), transparent 24rem),
          radial-gradient(circle at 95% 15%, rgba(255,219,193,.24), transparent 25rem),
          linear-gradient(180deg,#FAFAF4 0%,#F5F7EF 55%,#FBF8F0 100%);
        color:var(--ink);
      }
      .main .block-container { max-width:1460px; padding:1.55rem 2.1rem 3.1rem; }
      h1,h2,h3,h4 { font-family:'Nunito','DM Sans',sans-serif; letter-spacing:-.035em; color:var(--ink); }
      p { line-height:1.62; }

      /* Sidebar: friendly forest-green product navigation. */
      [data-testid="stSidebar"] {
        background:linear-gradient(165deg,#174B3A 0%,#205C43 54%,#174435 100%);
        border-right:1px solid rgba(255,255,255,.08);
      }
      [data-testid="stSidebar"] > div:first-child { padding-top:1.25rem; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li { color:#EAF4EA !important; }
      [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.18); margin:1.1rem 0; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label { color:#F0F6EC !important; font-weight:700; }
      [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + div { color:#D8F69C !important; }
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#BBD1C0 !important; text-transform:uppercase; font-size:.67rem; letter-spacing:1.35px; font-weight:800; }
      .brand { display:flex; align-items:center; gap:9px; color:#F8FFF7; font-family:'Nunito',sans-serif; font-size:1.65rem; font-weight:900; letter-spacing:-1px; }
      .brand span:last-child { color:#D4F28F; }
      .brand-mark { width:35px; height:35px; display:inline-flex; align-items:center; justify-content:center; border-radius:12px; background:linear-gradient(145deg,#D7F59B,#92D36B); color:#174B3A; font-size:1.2rem; box-shadow:0 4px 0 rgba(6,37,26,.28),0 7px 14px rgba(0,0,0,.15); }
      .side-note { color:#C5D7C9; font-size:.8rem; line-height:1.55; }
      .side-chip { display:inline-flex; align-items:center; gap:7px; border:1px solid rgba(215,245,155,.30); background:rgba(215,245,155,.10); color:#E0F7AC; border-radius:99px; padding:6px 11px; font-size:.66rem; font-weight:900; letter-spacing:.55px; }
      .side-group { display:flex; align-items:center; gap:8px; padding:8px 10px; margin:6px 0; border-radius:12px; background:rgba(255,255,255,.075); color:#F7FCF4; font-size:.8rem; font-weight:700; border:1px solid rgba(255,255,255,.07); }

      /* Playful 3D hero. */
      .hero {
        position:relative; overflow:hidden; display:flex; justify-content:space-between; align-items:center;
        gap:15px; background:linear-gradient(125deg,#E7F5E5 0%,#DDF0E7 48%,#F9EEDC 100%);
        border:1px solid #DDE9D8; border-radius:30px; padding:27px 30px 21px;
        margin:2px 0 20px; min-height:300px;
        box-shadow:0 7px 0 #DCE9D9,0 18px 34px rgba(49,91,63,.10);
      }
      .hero:before { content:''; position:absolute; width:210px; height:210px; border-radius:50%; background:rgba(255,255,255,.38); right:34%; top:-128px; pointer-events:none; }
      .hero:after { content:''; position:absolute; width:130px; height:130px; border-radius:50%; background:rgba(255,255,255,.25); left:-54px; bottom:-83px; pointer-events:none; }
      .hero-main { position:relative; z-index:2; flex:1 1 51%; max-width:610px; padding:7px 0 10px; }
      .hero-kicker { display:inline-flex; align-items:center; gap:8px; color:#39724F; text-transform:uppercase; letter-spacing:1.35px; font-size:.68rem; font-weight:900; background:rgba(255,255,255,.67); border:1px solid rgba(82,139,88,.14); border-radius:99px; padding:7px 11px; }
      .hero-kicker:before { content:'✦'; color:#77A83E; font-size:.8rem; }
      .hero h1 { font-family:'Nunito',sans-serif; font-size:clamp(2.1rem,3.5vw,3.05rem); line-height:1.04; letter-spacing:-1.9px; margin:.85rem 0 .72rem; color:#174B3A; font-weight:900; }
      .hero h1 span { color:#2D8A5E; }
      .hero p { color:#476456; font-size:.94rem; max-width:590px; margin:0; }
      .hero-tags { display:flex; flex-wrap:wrap; gap:8px; margin-top:19px; }
      .hero-tags span { color:#315746; border:1px solid rgba(83,131,83,.13); background:rgba(255,255,255,.77); border-radius:99px; padding:7px 11px; font-size:.71rem; font-weight:800; box-shadow:0 3px 0 rgba(184,211,178,.55); }
      .hero-visual { position:relative; z-index:1; flex:0 1 47%; min-width:300px; display:flex; align-items:center; justify-content:center; }
      .hero-illustration { width:100%; max-height:306px; object-fit:contain; display:block; filter:drop-shadow(0 13px 15px rgba(34,73,47,.13)); }
      .hero-sticker { position:absolute; right:1%; bottom:1%; transform:rotate(-4deg); background:#FFF5CE; color:#6A6336; border:2px solid #FFFDF1; border-radius:13px; padding:8px 12px; font-size:.69rem; font-weight:900; box-shadow:0 5px 0 #E4D7AB,0 9px 16px rgba(71,72,35,.12); }

      /* Three-step workflow cards. */
      .workflow { display:grid; grid-template-columns:repeat(3,1fr); gap:13px; margin:0 0 25px; }
      .workflow-item { display:flex; align-items:center; gap:12px; padding:13px 15px; background:#FFFFFF; border:1px solid var(--line); border-radius:18px; box-shadow:0 5px 0 #E7EBDD,0 9px 17px rgba(43,72,46,.055); transition:transform .18s ease,box-shadow .18s ease; }
      .workflow-icon { flex:0 0 42px; width:42px; height:42px; display:flex; align-items:center; justify-content:center; border-radius:14px; background:linear-gradient(145deg,#F0F8D9,#DDEFD8); font-size:1.32rem; box-shadow:inset 0 2px 3px rgba(255,255,255,.85),0 3px 0 #D5E5C9; }
      .workflow-item:nth-child(2) .workflow-icon { background:linear-gradient(145deg,#E4F2FF,#D4E7F6); box-shadow:inset 0 2px 3px rgba(255,255,255,.9),0 3px 0 #C9DCEB; }
      .workflow-item:nth-child(3) .workflow-icon { background:linear-gradient(145deg,#FFEAD9,#F8DCC9); box-shadow:inset 0 2px 3px rgba(255,255,255,.9),0 3px 0 #EED1BD; }
      .workflow-item b { display:block; font-size:.81rem; color:#264B3A; font-weight:900; }
      .workflow-item small { display:block; color:#839087; font-size:.69rem; line-height:1.35; }

      /* Panels, upload, results and category cards. */
      .section-title { font-family:'Nunito',sans-serif; font-weight:900; font-size:1.2rem; color:#1D4937; margin:0 0 2px; letter-spacing:-.35px; }
      .section-subtitle { color:#829087; font-size:.79rem; margin:0 0 13px; }
      .step-heading { display:flex; align-items:center; gap:11px; margin:0 0 13px; }
      .step-number { flex:0 0 38px; width:38px; height:38px; display:flex; align-items:center; justify-content:center; border-radius:14px; background:linear-gradient(145deg,#DDF1CE,#BFE49A); color:#285E3D; font-size:.84rem; font-weight:900; box-shadow:0 4px 0 #B4D48F,0 8px 13px rgba(49,94,54,.10); }
      .panel { background:linear-gradient(155deg,#FFFFFF 0%,#FDFEF9 100%); border:1px solid #E4EBDD; border-radius:23px; padding:21px; box-shadow:0 6px 0 #E8ECDF,0 13px 25px rgba(41,73,47,.055); }
      .muted { color:#718078; font-size:.88rem; }
      .result-card { position:relative; overflow:hidden; background:linear-gradient(145deg,#FFFFFF 0%,#F0F8ED 100%); border:1px solid #DAE8D7; border-radius:25px; padding:24px; box-shadow:0 7px 0 #DCE9D5,0 15px 27px rgba(36,76,46,.075); }
      .result-card:after { content:'✦'; position:absolute; right:19px; top:11px; color:rgba(130,174,97,.34); font-size:1.7rem; }
      .result-label { color:#7B8A81; font-size:.68rem; font-weight:900; text-transform:uppercase; letter-spacing:1.35px; }
      .result-name { color:#174B3A; font-family:'Nunito',sans-serif; font-weight:900; font-size:2.05rem; line-height:1.2; margin:8px 0 9px; }
      .pill { display:inline-flex; align-items:center; background:#EAF5D9; color:#37633E; border:1px solid #DDEBC7; border-radius:99px; padding:7px 12px; font-size:.76rem; font-weight:800; }
      .guidance { background:linear-gradient(110deg,#F1F8E8,#F9F7E9); border:1px solid #E0EACF; border-left:5px solid #8BBE56; border-radius:15px; padding:14px 16px; color:#385A40; line-height:1.58; font-size:.88rem; }
      .footnote { color:#879187; font-size:.74rem; line-height:1.58; background:rgba(255,255,255,.58); border:1px solid #EAEDE4; border-radius:13px; padding:11px 13px; }
      .group-card { border-radius:20px; padding:18px; background:linear-gradient(150deg,#FFFFFF 0%,#F7FAF1 100%); border:1px solid #E4EBDD; min-height:150px; box-shadow:0 5px 0 #E5EBDD,0 11px 20px rgba(20,55,37,.05); transition:transform .18s ease; }
      .group-card:hover { transform:translateY(-3px); }
      .group-icon { display:inline-flex; width:48px; height:48px; align-items:center; justify-content:center; border-radius:16px; background:linear-gradient(145deg,#EFF8D9,#DCEFCB); font-size:1.7rem; box-shadow:0 4px 0 #D0E5B9; margin-bottom:10px; }
      div[data-testid="stMetric"] { background:linear-gradient(145deg,#FFFFFF,#F5F9EF); border:1px solid #E3EBDD; padding:15px 16px; border-radius:18px; box-shadow:0 4px 0 #E6ECDD,0 8px 16px rgba(20,55,37,.035); }
      div[data-testid="stMetric"] label { color:#7B8A81 !important; font-size:.73rem !important; font-weight:800 !important; }
      div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#1D4937; font-family:'Nunito',sans-serif; font-weight:900; }
      .stButton button[kind="primary"] { background:linear-gradient(100deg,#217B55,#32956A); color:#FFFFFF; border:0; border-radius:15px; font-family:'Nunito',sans-serif; font-weight:900; padding:.78rem 1.2rem; min-height:48px; box-shadow:0 5px 0 #176344,0 9px 17px rgba(35,116,78,.18); transition:all .18s ease; }
      .stButton button[kind="primary"]:hover { background:linear-gradient(100deg,#1B6B49,#27845C); border:0; transform:translateY(-2px); box-shadow:0 7px 0 #176344,0 12px 20px rgba(35,116,78,.2); }
      .stDownloadButton button { border-radius:13px; font-weight:800; }
      [data-testid="stFileUploader"] { background:linear-gradient(145deg,#F7FBEF,#EFF7EA); border:2px dashed #B8D29F; border-radius:18px; padding:14px; }
      [data-testid="stFileUploader"] section { border:0; background:transparent; }
      [data-testid="stFileUploader"] button { border-radius:11px; font-weight:800; border:1px solid #D5E6C7; background:#FFFFFF; }
      [data-testid="stCameraInput"] { border-radius:17px; overflow:hidden; }
      [data-testid="stImage"] img { border-radius:18px; box-shadow:0 5px 0 #E7EBDD,0 12px 20px rgba(33,60,40,.08); }
      [data-testid="stRadio"] > div { gap:10px; }
      [data-testid="stRadio"] label { font-weight:700; color:#355443; }
      [data-testid="stAlert"] { border-radius:15px; border-width:1px; }
      [data-testid="stDataFrame"] { border:1px solid #E2EAE2; border-radius:15px; overflow:hidden; }
      .smallcaps { text-transform:uppercase; letter-spacing:1.3px; color:#6E8174; font-size:.68rem; font-weight:900; }
      @media (max-width:1050px) { .hero { padding:24px; } .hero-visual { min-width:235px; } .hero-illustration { max-height:255px; } }
      @media (max-width:850px) { .hero { flex-direction:column; align-items:flex-start; padding:23px 20px 13px; } .hero-main { max-width:none; } .hero-visual { width:100%; min-width:0; max-width:440px; align-self:center; } .hero-illustration { max-height:245px; } .hero-sticker { right:2%; bottom:0; } .workflow { grid-template-columns:1fr; gap:9px; } .main .block-container { padding:1.05rem 1rem 2rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[Any, Any]:
    """Load and cache the same pretrained classifier and image processor."""
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SiglipForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def predict_image(image: Image.Image, processor: Any, model: Any) -> list[dict[str, Any]]:
    """Run inference and read class labels from the pretrained model config."""
    rgb_image = image.convert("RGB")
    inputs = processor(images=rgb_image, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    model_labels = getattr(model.config, "id2label", {}) or {}
    results: list[dict[str, Any]] = []
    for index, probability in enumerate(probabilities):
        label = model_labels.get(index, model_labels.get(str(index), f"class_{index}"))
        label = str(label).strip().lower().replace(" ", "_").replace("/", "_")
        results.append({"category": label, "confidence": float(probability)})

    return sorted(results, key=lambda item: item["confidence"], reverse=True)


def get_sorting_details(category: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return material details and the three-way sorting recommendation."""
    if category not in WASTE_INFO or category not in CATEGORY_ROUTING:
        raise ValueError(
            f"The model returned an unsupported label: {category}. "
            "Check the model's id2label configuration before using this category."
        )
    return WASTE_INFO[category], CATEGORY_ROUTING[category]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">♻</span> Eco<span>Sort</span></div>', unsafe_allow_html=True)
        st.markdown(
            "<div class='side-note'>Smart Waste Classification<br>& Recycling Assistant</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br><span class='side-chip'>●  AI ASSISTANT READY</span>", unsafe_allow_html=True)
        st.divider()
        st.markdown("<div class='smallcaps'>Workspace</div>", unsafe_allow_html=True)
        page = st.radio(
            "NAVIGATION",
            ["Classify waste", "Project overview", "History"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("<div class='smallcaps'>Model information</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='side-note'>Pretrained 10-class image classifier.<br>Model files are cached locally after the first download.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br><div class='smallcaps'>Sorting destinations</div>", unsafe_allow_html=True)
        for group_name, group in SORTING_GROUPS.items():
            st.markdown(
                f"<div class='side-group'><span>{group['icon']}</span><span>{group_name}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<br><div class='side-note'>🔋 Batteries need authorised special collection. Never place them in ordinary bins.</div>",
            unsafe_allow_html=True,
        )
    return page


def render_header() -> None:
    """Render the friendly 3D hero and the three-step workflow."""
    illustration_path = Path(__file__).resolve().parent / "assets" / "ecosort_hero_3d.png"
    illustration_tag = ""
    try:
        if illustration_path.is_file():
            image_base64 = base64.b64encode(illustration_path.read_bytes()).decode("ascii")
            illustration_tag = (
                f'<img class="hero-illustration" src="data:image/png;base64,{image_base64}" '
                'alt="Three 3D illustrated waste-sorting bins with example waste items">'
            )
    except OSError:
        illustration_tag = ""

    if not illustration_tag:
        illustration_tag = '<div class="hero-fallback">♻️　🍃　🗑️</div>'

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-main">
            <div class="hero-kicker">Your friendly AI waste-sorting guide</div>
            <h1>Give every item<br><span>a better next step.</span></h1>
            <p>Snap a photo, discover the material, and find out whether it belongs in Recyclable, Organic or Residual waste.</p>
            <div class="hero-tags"><span>✦ 10 material labels</span><span>♻️ 3 sorting groups</span><span>📷 Photo or camera</span></div>
          </div>
          <div class="hero-visual">
            {illustration_tag}
            <div class="hero-sticker">Small actions. Greener habits.</div>
          </div>
        </div>
        <div class="workflow">
          <div class="workflow-item"><div class="workflow-icon">📸</div><div><b>1. Add a photo</b><small>Upload or capture one clear item</small></div></div>
          <div class="workflow-item"><div class="workflow-icon">🔎</div><div><b>2. Identify material</b><small>AI predicts the likely class</small></div></div>
          <div class="workflow-item"><div class="workflow-icon">🌱</div><div><b>3. Sort responsibly</b><small>Get a practical next step</small></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def classify_page() -> None:
    render_header()
    left, right = st.columns([1.02, 1.1], gap="large")

    with left:
        st.markdown("<div class='step-heading'><span class='step-number'>01</span><div><div class='section-title'>Add a waste item</div><div class='section-subtitle'>Choose a clear photo or capture one with your camera.</div></div></div>", unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        input_mode = st.radio(
            "Choose image source", ["Upload a photo", "Use camera"], horizontal=True
        )
        uploaded = None
        if input_mode == "Upload a photo":
            uploaded = st.file_uploader(
                "Choose one image",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )
        else:
            uploaded = st.camera_input("Take a clear photo of one waste item")
        st.markdown(
            "<p class='muted'>For the best result, photograph one item at a time in good light. Mixed-waste photos may receive only one overall prediction.</p>",
            unsafe_allow_html=True,
        )
        run_prediction = st.button(
            "♻️  Classify and sort",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        image = None
        if uploaded is not None:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Image selected for analysis", use_container_width=True)
            except (UnidentifiedImageError, OSError):
                st.error("This image could not be opened. Please choose a JPG, PNG or WebP image.")

    with right:
        st.markdown("<div class='step-heading'><span class='step-number'>02</span><div><div class='section-title'>Your sorting result</div><div class='section-subtitle'>Material identification and recommended next step.</div></div></div>", unsafe_allow_html=True)

        if run_prediction and image is not None:
            # Remove an older result so an unsuccessful new run cannot leave stale output visible.
            st.session_state.pop("latest_prediction", None)
            try:
                with st.spinner(
                    "Loading the AI model and analysing your image… The first run may take a little longer."
                ):
                    processor, model = load_model()
                    predictions = predict_image(image, processor, model)
                best = predictions[0]
                needs_review, review_reason, _ = assess_prediction(predictions)
                material_info = WASTE_INFO.get(
                    best["category"],
                    {"title": str(best["category"]).replace("_", " ").title()},
                )
                confidence_pct = best["confidence"] * 100

                # Do not force an unreliable prediction into Recyclable, Organic or Residual.
                # "Needs review" is a review status, not a fourth waste bin.
                if needs_review:
                    sorting_group_for_history = "Needs review"
                else:
                    _, sorting_info = get_sorting_details(best["category"])
                    sorting_group_for_history = sorting_info["group"]

                result_record = {
                    "Time": datetime.now().strftime("%d-%b-%Y %I:%M %p"),
                    "Image": uploaded.name if getattr(uploaded, "name", None) else "Camera image",
                    "Detected material": material_info["title"],
                    "Sorting group": sorting_group_for_history,
                    "Model score (%)": round(confidence_pct, 1),
                }
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.insert(0, result_record)
                st.session_state.latest_prediction = predictions
            except Exception as exc:
                st.error("The AI model could not be loaded or the image could not be analysed.")
                st.markdown(
                    "**Try this:** check your internet connection and restart the app. "
                    "If the problem continues, follow the troubleshooting section in `README.md`."
                )
                with st.expander("Technical details"):
                    st.code(str(exc))

        if "latest_prediction" in st.session_state:
            predictions = st.session_state.latest_prediction
            best = predictions[0]
            needs_review, review_reason, margin = assess_prediction(predictions)
            material_info = WASTE_INFO.get(
                best["category"],
                {"title": str(best["category"]).replace("_", " ").title(), "icon": "♻️"},
            )
            confidence_pct = best["confidence"] * 100

            if needs_review:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="result-label">Sorting status</div>', unsafe_allow_html=True)
                st.markdown(
                    "<div class='result-name'>⚠️ Mixed / unclear — needs review</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<span class='pill'>Top model guess: {material_info['title']} ({confidence_pct:.1f}%)</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                st.warning(review_reason)
                st.markdown(
                    "**Recommended next step:** separate the pile and upload one item at a time. "
                    "Do not treat this uncertain result as a confirmed Recyclable, Organic or Residual decision.",
                )
                st.info(
                    "A mixed photograph can contain paper, clothes, plastic, metal and other materials. "
                    "This classifier returns one overall label; it does not detect and sort every object in the scene."
                )
                if best["category"] == "battery" or any(
                    item["category"] == "battery" for item in predictions[:3]
                ):
                    st.error(
                        "Safety reminder: if the image contains a battery, keep it separate and take it "
                        "to an authorised battery collection point. Do not place it in an ordinary bin."
                    )
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                material_info, sorting_info = get_sorting_details(best["category"])
                group_name = sorting_info["group"]
                group_info = SORTING_GROUPS[group_name]

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="result-label">Primary sorting category</div>', unsafe_allow_html=True)
                st.markdown(
                    f"<div class='result-name'>{group_info['icon']} {group_name}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<span class='pill'>Material identified: {material_info['title']}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="result-label">Suggested sorting stream</div>', unsafe_allow_html=True)
                st.markdown(f"**{sorting_info['stream']}**")
                st.markdown(
                    f"<div class='guidance'>{sorting_info['guidance']}</div>",
                    unsafe_allow_html=True,
                )
                if best["category"] == "battery":
                    st.error(sorting_info["warning"])
                else:
                    st.info(sorting_info["warning"])
                st.markdown("</div>", unsafe_allow_html=True)

            score_col, material_col = st.columns(2)
            score_col.metric("AI model score", f"{confidence_pct:.1f}%")
            material_col.metric(
                "Detected material",
                material_info["title"] if not needs_review else f"Uncertain: {material_info['title']}",
            )

            chart_data = pd.DataFrame(
                [
                    {
                        "Material category": WASTE_INFO[item["category"]]["title"],
                        "Model score": item["confidence"] * 100,
                    }
                    for item in predictions
                ]
            )
            st.markdown("<br><div class='section-title'>AI model scores by material</div>", unsafe_allow_html=True)
            fig = px.bar(
                chart_data,
                x="Model score",
                y="Material category",
                orientation="h",
                text=chart_data["Model score"].map(lambda value: f"{value:.1f}%"),
            )
            bar_colors = [WASTE_INFO[item["category"]]["colour"] for item in predictions]
            fig.update_traces(marker_color=bar_colors, marker_line_width=0, textposition="outside", cliponaxis=False, textfont=dict(size=12, color="#355443"))
            fig.update_layout(
                height=335,
                margin=dict(l=5, r=42, t=12, b=10),
                xaxis_title="Model score (%)",
                yaxis_title="",
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#355443", family="DM Sans, sans-serif", size=12),
                xaxis=dict(range=[0, 108], gridcolor="#E7EEE8"),
                yaxis=dict(categoryorder="total ascending"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "The model score is a softmax output, not a calibrated probability that the prediction is correct. A high score can still be wrong."
            )
        else:
            st.markdown(
                "<div class='panel' style='min-height:330px;display:flex;flex-direction:column;justify-content:center;text-align:center;'>"
                "<div style='font-size:3.2rem;margin-bottom:8px;'>🌱</div><div class='result-label'>Ready when you are</div><h3 style='color:#244B36;margin:.65rem 0;'>Your sorting result will appear here</h3>"
                "<p class='muted'>Upload one clear waste item and select <b>Classify and sort</b>. EcoSort will identify the material and recommend a sorting group.</p>"
                "<div style='display:flex;justify-content:center;gap:7px;flex-wrap:wrap;margin-top:10px;'><span class='pill'>♻️ Recyclable</span><span class='pill'>🍃 Organic</span><span class='pill'>🗑️ Residual</span></div>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='footnote'>Educational prototype. The AI predicts one label for the whole image; it does not separate multiple objects in a mixed-waste photograph. Low or closely tied model scores are marked “Mixed / unclear — needs review” instead of being forced into a bin. The three-way sorting group is a rule-based recommendation built on top of a sufficiently clear prediction. Local collection rules vary. Batteries always require a separate authorised collection route.</div>",
        unsafe_allow_html=True,
    )


def overview_page() -> None:
    render_header()
    st.markdown("## About EcoSort")
    st.write(
        "EcoSort is an educational computer-vision application. It uses a pretrained image classifier to identify the likely material or object in a photo, then applies a transparent rule-based mapping to recommend one of three main sorting groups: Recyclable, Organic or Residual."
    )

    st.markdown("## The three sorting groups")
    group_cols = st.columns(3)
    for column, (group_name, info) in zip(group_cols, SORTING_GROUPS.items()):
        with column:
            st.markdown(
                f"<div class='group-card'><div class='group-icon'>{info['icon']}</div><br><b>{group_name}</b><p class='muted'>{info['description']}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("## Material classes recognised by the model")
    category_cols = st.columns(3)
    for index, (category, info) in enumerate(WASTE_INFO.items()):
        sorting_group = CATEGORY_ROUTING[category]["group"]
        with category_cols[index % 3]:
            st.markdown(
                f"<div class='panel' style='margin-bottom:12px;'><span style='font-size:1.6rem'>{info['icon']}</span><br><b>{info['title']}</b><br><span class='muted'>Sorting group: {sorting_group}</span></div>",
                unsafe_allow_html=True,
            )

    st.warning(
        "Special case: batteries are shown under the Residual branch for the three-way overview, but they must NOT go into an ordinary residual bin. Take them to an authorised battery collection point."
    )

    st.markdown("## How the system works")
    steps = [
        "The user uploads or captures a photo of one waste item.",
        "Pillow converts the image to RGB and the model processor prepares it.",
        "The pretrained SigLIP2 image classifier predicts one material/object label.",
        "The app reads the label and score from the model output.",
        "A confidence and top-two-score check flags uncertain or mixed photos for review instead of forcing a bin choice.",
        "For sufficiently clear predictions, a rule-based mapping assigns Recyclable, Organic or Residual and displays sorting guidance.",
    ]
    for index, step in enumerate(steps, start=1):
        st.markdown(f"**{index}.** {step}")

    st.markdown("## Technologies used")
    tech = pd.DataFrame(
        [
            ["Python", "Main programming language"],
            ["Streamlit", "Web interface and user interaction"],
            ["PyTorch", "Model inference"],
            ["Hugging Face Transformers", "Load the pretrained image processor and classifier"],
            ["Pillow", "Open and convert uploaded images"],
            ["Pandas + Plotly", "Prediction history and model-score chart"],
        ],
        columns=["Technology", "Role in the project"],
    )
    st.dataframe(tech, use_container_width=True, hide_index=True)

    st.info(
        "Model attribution: prithivMLmods/Augmented-Waste-Classifier-SigLIP2. The pretrained model supplies the material/object prediction; EcoSort adds the three-way sorting recommendation using its own rule-based mapping."
    )
    st.warning(
        "The pretrained model is third-party software integrated into this application; this project has not independently trained or benchmarked it. Model scores are not verified accuracy. The model predicts one class for the whole image and may perform poorly on mixed-waste photos. Low or closely tied predictions are shown as “Mixed / unclear — needs review”."
    )


def history_page() -> None:
    render_header()
    st.markdown("## Your current session")
    history = st.session_state.get("history", [])
    if not history:
        st.info("No predictions yet. Go to **Classify waste** and analyse an image to build your session history.")
        return

    df = pd.DataFrame(history)
    col1, col2, col3 = st.columns(3)
    col1.metric("Images analysed", len(df))
    col2.metric("Sorting groups found", df["Sorting group"].nunique())
    col3.metric("Average model score", f"{df['Model score (%)'].mean():.1f}%")
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download session history (CSV)",
        data=csv_bytes,
        file_name="ecosort_prediction_history.csv",
        mime="text/csv",
    )
    if st.button("Clear session history"):
        st.session_state.history = []
        st.session_state.pop("latest_prediction", None)
        st.rerun()


page = render_sidebar()
if page == "Classify waste":
    classify_page()
elif page == "Project overview":
    overview_page()
else:
    history_page()
