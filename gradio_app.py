#!/usr/bin/env python3
"""
MedSAM3 — webové rozhraní (Gradio). Model běží na AWS serveru; rozhraní
posílá vstupy a zobrazuje výstupy.

Záložky:
  • Co umí           – přehled schopností technologie
  • Obecná medicína  – běžné radiologické / klinické případy
  • Výzkum (TNT/PLA) – cíl projektu: TNT, PLA tečky, buňky
  • Stav projektu    – vývojový log
"""
import time
import tempfile

from PIL import Image
import gradio as gr

from infer_sam import SAM3LoRAInference
from tiling import tiled_predict

CONFIG = "configs/full_lora_config.yaml"
WEIGHTS = "weights/medsam3_v1/best_lora_weights.pt"

# --- pravidla pro vstupní snímek (best practices) ---
TILE = 1008            # nativní vstup SAM3
MIN_SIDE = 600         # pod tím nemá model dost detailů -> odmítnout
MAX_SIDE = 4000        # nad tím zmenšíme, aby výpočet (dlaždice na CPU) nebyl extrémní
TILE_TRIGGER = 1300    # automaticky dlaždicovat, když delší strana přesáhne tuto hodnotu
MAX_MEGAPIXELS = 40    # tvrdý strop (ochrana paměti)

# ---- knihovny promptů (český popisek -> anglický prompt, model rozumí anglicky) ----
GENERAL_PROMPTS = {
    "Plíce – RTG (lung)": "lung",
    "Plicní uzel (lung nodule)": "lung nodule",
    "Zlomenina kosti (bone fracture)": "bone fracture",
    "Nádor / léze (tumor)": "tumor",
    "Játra – CT (liver)": "liver",
    "Ledvina (kidney)": "kidney",
    "Kožní léze – dermatoskopie (skin lesion)": "skin lesion",
    "Polyp – endoskopie (polyp)": "polyp",
    "Buněčné jádro – histologie (cell nucleus)": "cell nucleus",
}

PLA_PROMPTS = {
    "PLA tečka (fluorescent spot)": "fluorescent spot",
    "PLA puncta (fluorescent puncta)": "fluorescent puncta",
    "Jádro buňky – ROI (nucleus)": "nucleus",
    "Buňka – ROI (cell)": "cell",
}

# více variant pro TNT — testujeme a učíme se, která formulace funguje nejlépe
TNT_VARIANTS = {
    "tunneling nanotube": "tunneling nanotube",
    "thin tube connecting cells": "thin tube connecting cells",
    "membrane bridge between cells": "membrane bridge between cells",
    "intercellular bridge": "intercellular bridge",
    "thin membrane protrusion": "thin membrane protrusion",
    "filament between cells": "filament between cells",
    "actin filament": "actin filament",
}

print("⏳ Načítám model (jednorázově)…", flush=True)
ENGINE = SAM3LoRAInference(config_path=CONFIG, weights_path=WEIGHTS,
                           detection_threshold=0.4, nms_iou_threshold=0.5)
print("✅ Model připraven.", flush=True)


def validate_image(image):
    """Validátor vstupu. Vrací (pripraveny_obrazek | None, chyba, poznamka)."""
    if image is None:
        return None, "⚠️ Nejprve nahrajte snímek.", ""
    try:
        img = image.convert("RGB")
    except Exception:
        return None, "⚠️ Soubor nelze načíst jako obrázek.", ""

    W, H = img.size
    if min(W, H) < 16:
        return None, "⚠️ Snímek má neplatné rozměry.", ""
    if max(W, H) < MIN_SIDE:
        return None, (f"⚠️ Snímek je příliš malý ({W}×{H} px). "
                      f"Minimální delší strana je **{MIN_SIDE} px** — nahrajte snímek "
                      f"ve vyšším rozlišení (zvětšování zde nepomáhá, model jinak "
                      f"nemá dostatek skutečných detailů)."), ""
    if (W * H) / 1e6 > MAX_MEGAPIXELS or max(W, H) > MAX_SIDE:
        s = min(MAX_SIDE / max(W, H), (MAX_MEGAPIXELS * 1e6 / (W * H)) ** 0.5)
        img = img.resize((max(1, int(W * s)), max(1, int(H * s))), Image.LANCZOS)
        return img, "", f"snímek zmenšen na {img.size[0]}×{img.size[1]} px (limit výkonu)"

    ratio = max(W, H) / min(W, H)
    note = "⚠️ velmi protáhlý poměr stran" if ratio > 6 else ""
    return img, "", note


def analyze(image, prompts, threshold, nms, show_boxes, mode):
    img, err, info = validate_image(image)
    if err:
        return None, err
    if not prompts:
        return None, "⚠️ Vyberte alespoň jeden cíl (nebo zadejte vlastní)."

    W, H = img.size
    if mode.startswith("Jeden"):
        use_tiling = False
    elif mode.startswith("Vždy"):
        use_tiling = True
    else:  # Automaticky – best practice: dlaždice jen u velkých snímků
        use_tiling = max(W, H) > TILE_TRIGGER

    ENGINE.detection_threshold = float(threshold)
    ENGINE.nms_iou_threshold = float(nms)

    t0 = time.time()
    if use_tiling:
        results = tiled_predict(ENGINE, img, prompts, tile=TILE, overlap=0.25)
        how = "dlaždice"
    else:
        tmp_in = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        img.save(tmp_in)
        results = ENGINE.predict(tmp_in, prompts)
        how = "jeden průchod"
    tmp_out = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    ENGINE.visualize(results, tmp_out, show_boxes=show_boxes, show_masks=True)
    dt = time.time() - t0

    meta = f"{how} · {W}×{H} px"
    if info:
        meta += f" · {info}"
    lines = [f"### Výsledky  (⏱️ {dt:.0f} s · {meta})", ""]
    total = 0
    for idx in sorted([k for k in results if k != "_image"]):
        r = results[idx]
        n = r["num_detections"]
        total += n
        if n:
            lines.append(f"- **{r['prompt']}**: {n}  ·  max. jistota {float(r['scores'].max()):.2f}")
        else:
            lines.append(f"- _{r['prompt']}_: 0")
    lines += ["", f"**Celkem nalezeno: {total}**"]
    return Image.open(tmp_out), "\n".join(lines)


def _controls():
    """Sdílené ovládací prvky."""
    mode = gr.Radio(
        ["Automaticky (doporučeno)", "Vždy dlaždice", "Jeden průchod"],
        value="Automaticky (doporučeno)",
        label="Režim zpracování (dlaždice = přesnější u velkých snímků, pomalejší)")
    with gr.Accordion("Pokročilé nastavení", open=False):
        threshold = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                              label="Práh jistoty (nižší = více nálezů)")
        nms = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                        label="NMS IoU (nižší = méně překryvů)")
        show_boxes = gr.Checkbox(value=True, label="Zobrazit ohraničení (rámečky)")
    return threshold, nms, show_boxes, mode


CAPABILITIES_MD = """
# 🧠 MedSAM3 — segmentace obrazu pomocí lékařských pojmů

**Co to umí:** místo ručního klikání nebo kreslení rámečků prostě **slovem popíšete,
co hledat**, a model to v obraze **najde, vyznačí (maska) a spočítá**.

### Podporované modality
🩻 RTG · 🧠 CT / MRI / PET · 🔊 ultrazvuk · 🔬 mikroskopie & histopatologie ·
🩹 dermatoskopie · 🫁 endoskopie · 🧫 fluorescenční mikroskopie (buňky)

### K čemu je to dobré
- **Detekce & segmentace** orgánů, lézí, nádorů, buněk…
- **Počítání** objektů (např. PLA teček, jader, buněk)
- **Definice oblastí zájmu (ROI)** pro následné měření
- **Bez anotací** — řízeno textem (tzv. *concept-guided*)

> Postaveno na **Meta SAM3** + jemné doladění **LoRA** pro lékařské pojmy (MedSAM3).
> Vyberte si nahoře záložku **Obecná medicína**, **PLA** nebo **TNT**.
"""

STATUS_MD = """
# 📋 Stav projektu / vývojový log

**Cíl projektu:** automatizovat **detekci a měření tunelujících nanotrubic (TNT)**
a **počítání PLA teček s definicí ROI** ve fluorescenční mikroskopii — a u toho
využít MedSAM3 i pro obecné lékařské snímky.

### ✅ Hotovo
- Fork repozitáře, nasazení na **AWS** (CPU instance, eu-central-1)
- Zprovoznění **SAM3 + MedSAM3 LoRA** (CPU patche pro CUDA-only kód)
- České **webové rozhraní** (toto) jako trvalá služba (systemd)
- **PLA / jádra / buňky**: detekce funguje *zero-shot* (jistota 0,9+)
- **Optimalizace velkých snímků – dlaždice**: na testovacím PLA snímku
  tečky 13 → 43, jádra 31 → 68, buňky 25 → 49 (a vyšší jistota)
- **Validátor vstupu** + automatické rozhodnutí o dlaždicování podle velikosti
- Zjištěno: SAM3 je pevně vázán na vstup **1008×1008**; prosté **zvětšování
  snímku nepomáhá** (interpolace nepřidá detail) → odstraněno, řešením je dlaždicování

### 🔄 Probíhá
- **Prompt engineering** — profesionálnější formulace, více variant pro TNT (testujeme)
- Ladění velikosti dlaždic / překryvu

### ⏭️ Plánováno
- **Počítání PLA teček na jednu buňku** (přiřazení teček k ROI jádra)
- **TNT**: dotrénování (LoRA) na anotovaných snímcích — TNT není ve slovníku v1
- **GPU** instance (po navýšení kvóty) → ~20× rychlejší
- Stabilní (Elastic IP) odkaz pro sdílení

_Poslední aktualizace: 2026-06-15._
"""

TECH_MD = (
    "<hr><small><b>Tech stack:</b> Meta <b>SAM3</b> + <b>MedSAM3</b> LoRA · "
    "PyTorch 2.7 · Hugging Face · Gradio · běží na <b>AWS EC2</b> (eu-central-1) · "
    "Python/systemd. Model běží na serveru, nahrané snímky se neukládají. "
    "Zdroj: github.com/Boza-Analytics/MedSAM3</small>"
)


with gr.Blocks(title="MedSAM3") as demo:
    gr.Markdown("## 🔬 MedSAM3 — analýza lékařských a mikroskopických snímků")

    with gr.Tabs():
        # ---- 1) Co umí ----
        with gr.Tab("ℹ️ Co umí"):
            gr.Markdown(CAPABILITIES_MD)

        # ---- 2) Obecná medicína ----
        with gr.Tab("🩻 Obecná medicína"):
            gr.Markdown("Běžné případy: RTG, CT/MRI, dermatoskopie, endoskopie, histologie. "
                        "Vyberte cíl, nahrajte snímek a spusťte analýzu.")
            with gr.Row():
                with gr.Column():
                    g_img = gr.Image(type="pil", label="Vstupní snímek")
                    g_sel = gr.CheckboxGroup(list(GENERAL_PROMPTS.keys()),
                                             value=["Nádor / léze (tumor)"], label="Co hledat")
                    g_custom = gr.Textbox(label="Vlastní cíl(e) anglicky, oddělené čárkou",
                                          placeholder="např. spleen, pancreas")
                    g_thr, g_nms, g_box, g_mode = _controls()
                    g_btn = gr.Button("▶ Spustit analýzu", variant="primary")
                with gr.Column():
                    g_out = gr.Image(label="Výsledek", type="pil")
                    g_txt = gr.Markdown()
            g_btn.click(
                lambda im, sel, cu, t, n, b, m: analyze(
                    im, [GENERAL_PROMPTS[s] for s in (sel or [])]
                    + [p.strip() for p in (cu or "").split(",") if p.strip()],
                    t, n, b, m),
                [g_img, g_sel, g_custom, g_thr, g_nms, g_box, g_mode], [g_out, g_txt])

        # ---- 3) PLA ----
        with gr.Tab("🟢 PLA — počítání teček + ROI"):
            gr.Markdown(
                "**Proximity Ligation Assay.** Cíl: spočítat **PLA tečky** (každá = "
                "interakce proteinů) a vymezit **oblasti zájmu (ROI)** — jádra/buňky. "
                "Funguje dobře i bez dotrénování. U velkých snímků doporučujeme dlaždice "
                "(automaticky).")
            with gr.Row():
                with gr.Column():
                    p_img = gr.Image(type="pil", label="Vstupní snímek (fluorescenční)")
                    p_sel = gr.CheckboxGroup(
                        list(PLA_PROMPTS.keys()),
                        value=["PLA tečka (fluorescent spot)", "Jádro buňky – ROI (nucleus)"],
                        label="Co hledat")
                    p_custom = gr.Textbox(label="Vlastní cíl(e) anglicky, oddělené čárkou",
                                          placeholder="např. mitochondria")
                    p_thr, p_nms, p_box, p_mode = _controls()
                    p_btn = gr.Button("▶ Spustit analýzu", variant="primary")
                with gr.Column():
                    p_out = gr.Image(label="Výsledek", type="pil")
                    p_txt = gr.Markdown()
            p_btn.click(
                lambda im, sel, cu, t, n, b, m: analyze(
                    im, [PLA_PROMPTS[s] for s in (sel or [])]
                    + [p.strip() for p in (cu or "").split(",") if p.strip()],
                    t, n, b, m),
                [p_img, p_sel, p_custom, p_thr, p_nms, p_box, p_mode], [p_out, p_txt])

        # ---- 4) TNT ----
        with gr.Tab("🧵 TNT — tunelující nanotrubice"):
            gr.Markdown(
                "**Tunneling nanotubes.** Cíl: detekovat a měřit tenké trubice mezi "
                "buňkami. ⚠️ **Nejtěžší úloha** — TNT není ve slovníku modelu v1, takže "
                "*zero-shot* je slabý. Zde **testujeme více formulací promptu** (vyberte "
                "několik a porovnejte). Skutečné zlepšení přinese **dotrénování** "
                "(viz záložka Stav projektu).")
            with gr.Row():
                with gr.Column():
                    t_img = gr.Image(type="pil", label="Vstupní snímek")
                    t_tnt = gr.CheckboxGroup(
                        list(TNT_VARIANTS.keys()), value=["tunneling nanotube"],
                        label="Varianty promptu pro TNT (testovací — vyberte více a porovnejte)")
                    t_ctx = gr.Checkbox(value=True, label="Vyznačit i buňky pro kontext (cell)")
                    t_custom = gr.Textbox(label="Vlastní cíl(e) anglicky, oddělené čárkou",
                                          placeholder="např. nanotube")
                    t_thr, t_nms, t_box, t_mode = _controls()
                    t_btn = gr.Button("▶ Spustit analýzu", variant="primary")
                with gr.Column():
                    t_out = gr.Image(label="Výsledek", type="pil")
                    t_txt = gr.Markdown()
            t_btn.click(
                lambda im, tnt, ctx, cu, t, n, b, m: analyze(
                    im, [TNT_VARIANTS[x] for x in (tnt or [])]
                    + (["cell"] if ctx else [])
                    + [p.strip() for p in (cu or "").split(",") if p.strip()],
                    t, n, b, m),
                [t_img, t_tnt, t_ctx, t_custom, t_thr, t_nms, t_box, t_mode], [t_out, t_txt])

        # ---- 5) Stav projektu ----
        with gr.Tab("📋 Stav projektu"):
            gr.Markdown(STATUS_MD)

    gr.Markdown(TECH_MD)


if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=7860,
                                  share=False, show_error=True)
