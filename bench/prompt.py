"""The one canonical prompt every model receives. No per-model tuning (spec non-goal)."""
import os

SCHEMA = ('{"detections":[{"brand":"<name>","box":[x0,y0,x1,y1],'
          '"size":"small|medium|large","placement":"foreground|background",'
          '"location":"chest|sleeve|shorts|headwear|board|backdrop|other","conf":1|2|3}]}')

RETRY_SUFFIX = ("\n\nIMPORTANT: your previous answer was not parseable. "
                "Respond ONLY with the single-line JSON object, no prose, no code fences.")


def build_prompt(brands, n_refs=0):
    brand_lines = "\n".join(f'- "{b["name"]}": {b["description"]}' for b in brands)
    ref_part = ""
    if n_refs:
        ref_part = (f"The first {n_refs} images are REFERENCE crops showing what the brand "
                    "marks look like; they are NOT the image to analyze. The LAST image is "
                    "the TARGET image.\n\n")
    return (f"{ref_part}You are a sponsor-logo auditor. Find EVERY visible logo instance "
            f"of these brands in the TARGET image:\n{brand_lines}\n\n"
            "Rules:\n"
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


def load_refs(brands, root):
    """(jpeg_bytes, brand_name) for every configured reference image, brand order."""
    refs = []
    for b in brands:
        for rel in b.get("refs") or []:
            with open(os.path.join(root, rel), "rb") as f:
                refs.append((f.read(), b["name"]))
    return refs
