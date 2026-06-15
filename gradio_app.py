#!/usr/bin/env python3
"""
MedSAM3 – jednoduché webové rozhraní (Gradio) pro detekci a segmentaci
v mikroskopických snímcích. Model běží na tomto serveru (AWS), rozhraní
pouze posílá vstupy a zobrazuje výstupy.

Spuštění:  python3 gradio_app.py   (viz deploy/run_frontend.sh)
"""
import os
import time
import tempfile

from PIL import Image
import gradio as gr

from infer_sam import SAM3LoRAInference
from tiling import tiled_predict

CONFIG = "configs/full_lora_config.yaml"
WEIGHTS = "weights/medsam3_v1/best_lora_weights.pt"

# Český popisek -> anglický prompt (model rozumí angličtině)
PROMPT_MAP = {
    "Jádro buňky (nucleus)": "nucleus",
    "Buňka (cell)": "cell",
    "Fluorescenční tečka – PLA (fluorescent spot)": "fluorescent spot",
    "Tunelující nanotrubice – TNT (tunneling nanotube)": "tunneling nanotube",
    "Membránový výběžek (cell membrane protrusion)": "cell membrane protrusion",
}

print("⏳ Načítám model (jednorázově, může trvat ~1 min)…")
ENGINE = SAM3LoRAInference(
    config_path=CONFIG,
    weights_path=WEIGHTS,
    detection_threshold=0.4,
    nms_iou_threshold=0.5,
)
print("✅ Model připraven.")


def run(image, cz_prompts, custom_prompt, threshold, nms, show_boxes, use_tiling):
    if image is None:
        return None, "⚠️ Nejprve nahrajte obrázek."

    prompts = [PROMPT_MAP[p] for p in (cz_prompts or [])]
    if custom_prompt and custom_prompt.strip():
        prompts.append(custom_prompt.strip())
    if not prompts:
        return None, "⚠️ Vyberte alespoň jeden cíl, nebo zadejte vlastní (anglicky)."

    ENGINE.detection_threshold = float(threshold)
    ENGINE.nms_iou_threshold = float(nms)

    t0 = time.time()
    if use_tiling:
        # split the image into overlapping 1008 crops -> better on large images
        results = tiled_predict(ENGINE, image.convert("RGB"), prompts)
    else:
        tmp_in = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        image.convert("RGB").save(tmp_in)
        results = ENGINE.predict(tmp_in, prompts)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    ENGINE.visualize(results, tmp_out, show_boxes=show_boxes, show_masks=True)
    dt = time.time() - t0

    lines = [f"### Výsledky  (⏱️ {dt:.0f} s)", ""]
    total = 0
    for idx in sorted([k for k in results if k != "_image"]):
        r = results[idx]
        n = r["num_detections"]
        total += n
        if n:
            lines.append(f"- **{r['prompt']}**: {n} nálezů  ·  max. jistota {r['scores'].max():.2f}")
        else:
            lines.append(f"- **{r['prompt']}**: 0 nálezů")
    lines.append("")
    lines.append(f"**Celkem nalezeno: {total}**")
    return Image.open(tmp_out), "\n".join(lines)


# příklady ze složky s testovacími snímky
def _examples():
    ex = []
    for sub in ("pla", "tnt"):
        d = os.path.join("test_images", sub)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    ex.append([os.path.join(d, f)])
    return ex


with gr.Blocks(title="MedSAM3 – mikroskopie") as demo:
    gr.Markdown(
        """
        # 🔬 MedSAM3 — detekce a segmentace v mikroskopii
        Nástroj automaticky **najde a vyznačí** struktury v mikroskopickém snímku
        podle zadaného cíle. Cílem projektu je automatizace detekce **TNT
        (tunelujících nanotrubic)** a počítání **PLA teček** s vymezením oblastí (ROI).

        **Jak to použít:** 1) nahrajte snímek (nebo vyberte příklad níže) · 2) zvolte
        cíle · 3) klikněte na **Spustit analýzu**.
        > ⚙️ Model běží na CPU serveru, jeden výpočet trvá přibližně **1–2 minuty**.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Vstupní snímek")
            cz_prompts = gr.CheckboxGroup(
                choices=list(PROMPT_MAP.keys()),
                value=["Jádro buňky (nucleus)", "Buňka (cell)"],
                label="Co hledat",
            )
            custom_prompt = gr.Textbox(
                label="Vlastní cíl (anglicky, nepovinné)",
                placeholder="např. mitochondria",
            )
            with gr.Accordion("Pokročilé nastavení", open=False):
                threshold = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                      label="Práh jistoty (nižší = více nálezů)")
                nms = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                                label="NMS IoU (nižší = méně překryvů)")
                show_boxes = gr.Checkbox(value=True, label="Zobrazit ohraničení (rámečky)")
            use_tiling = gr.Checkbox(
                value=False,
                label="🔬 Režim velkého snímku (dlaždice) – přesnější u velkých snímků, ale výrazně pomalejší",
            )
            btn = gr.Button("▶ Spustit analýzu", variant="primary")
        with gr.Column(scale=1):
            out_img = gr.Image(label="Výsledek", type="pil")
            out_txt = gr.Markdown()

    gr.Examples(examples=_examples(), inputs=inp, label="Příklady (PLA a TNT)")

    btn.click(run, [inp, cz_prompts, custom_prompt, threshold, nms, show_boxes, use_tiling],
              [out_img, out_txt])

    gr.Markdown(
        "<small>MedSAM3 (SAM3 + LoRA) · zdrojové snímky CC-BY/PD viz "
        "<code>test_images/SOURCES.md</code> · model běží na AWS, data se nikam neukládají.</small>"
    )

if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # accessed directly via http://<EC2_IP>:7860 (port opened in SG)
        show_error=True,
    )
