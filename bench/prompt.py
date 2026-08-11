"""The one canonical prompt every model receives. No per-model tuning is a
spec non-goal; conditions (reference images, the zoom tool) are declared per
model row and change the prompt identically for every model in that condition."""
import os

SCHEMA = ('{"detections":[{"brand":"<name>","box":[x0,y0,x1,y1],'
          '"size":"small|medium|large","placement":"foreground|background",'
          '"location":"chest|sleeve|shorts|headwear|board|backdrop|other","conf":1|2|3}]}')

RETRY_SUFFIX = ("\n\nIMPORTANT: your previous answer was not parseable. "
                "Respond ONLY with the single-line JSON object, no prose, no code fences.")

ZOOM_LIMIT = 3

ZOOM_PART = (
    "If you need a closer look before answering, you may request an enlarged "
    "crop: respond ONLY with {\"zoom\": [x0,y0,x1,y1]} (integers 0-1000 over "
    f"the TARGET image). You may zoom up to {ZOOM_LIMIT} times in total; each "
    "zoom result arrives as a new image. When you are done looking, respond "
    "with the final detections JSON. Box coordinates in your final answer are "
    "ALWAYS relative to the full original TARGET image, never to a zoomed crop.\n\n")


def build_prompt(brands, ref_labels=(), zoom=False):
    brand_lines = "\n".join(f'- "{b["name"]}": {b["description"]}' for b in brands)
    ref_part = ""
    if ref_labels:
        numbered = " ".join(f"{i + 1}. {lb}." for i, lb in enumerate(ref_labels))
        ref_part = (f"The first {len(ref_labels)} images are REFERENCE images showing "
                    f"what the brand marks look like, in this order: {numbered} "
                    "They are NOT the image to analyze. The LAST image is the "
                    "TARGET image.\n\n")
    return (f"{ref_part}You are a sponsor-logo auditor. Find EVERY visible logo instance "
            f"of these brands in the TARGET image:\n{brand_lines}\n\n"
            + (ZOOM_PART if zoom else "")
            + "Rules:\n"
            "- Report each visible instance separately, including tiny, blurry, or partly "
            "occluded ones.\n"
            "- box: integers 0-1000 over the FULL target image, [x0,y0,x1,y1], tight around "
            "the mark.\n"
            "- size: small (you must squint to see it), medium, large (a dominant element "
            "of the frame).\n"
            "- placement: foreground (on a person or object that is the subject) or "
            "background (backdrops, boards, banners, out-of-focus areas).\n"
            "- location: chest|sleeve|shorts|headwear|board|backdrop|other.\n"
            "- conf: 1 unsure, 2 confident, 3 certain.\n"
            "- Only the listed brands. If none are visible, \"detections\" is [].\n\n"
            f"Respond ONLY with compact single-line JSON: {SCHEMA}")


def load_refs(brands, root, sheet_path=None):
    """(jpeg_bytes, label) for the optional labeled reference sheet followed by
    every configured per-brand reference image, in brand order."""
    refs = []
    if sheet_path:
        with open(os.path.join(root, sheet_path), "rb") as f:
            refs.append((f.read(), "a labeled reference sheet of all brand marks"))
    for b in brands:
        for rel in b.get("refs") or []:
            with open(os.path.join(root, rel), "rb") as f:
                refs.append((f.read(), f'a reference crop of the "{b["name"]}" mark'))
    return refs
