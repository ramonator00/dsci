# Verwendete Bibliotheken
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('<your directory>')
import object_extraction as oe
from matplotlib.colors import rgb_to_hsv as mpl_rgb_to_hsv
from pathlib import Path
from skimage.transform import resize
from skimage.morphology import remove_small_holes, remove_small_objects, square
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_opening, binary_closing, convex_hull_image, disk
from skimage.measure import label
from skimage.io import imsave


# ==========================================================
# KONFIGURATION
# ==========================================================

# Alle Bilder aus diesem Ordner verarbeiten
assets_path = Path('./assets')
if not assets_path.is_dir():
    raise FileNotFoundError(f'Assets-Ordner nicht gefunden: {assets_path}')

bild_endungen = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
imgfns = sorted(
    fn for fn in assets_path.rglob('*')
    if fn.is_file() and fn.suffix in bild_endungen
)

if not imgfns:
    raise FileNotFoundError(f'Keine Bilder in {assets_path} gefunden')

# Ausgabeordner
Ausgabepfad = Path('./')
Ausgabepfad.mkdir(exist_ok=True)

# Extraktionsparameter
pr = (225, 400)   # Auflösung für Vorverarbeitung
fr = 0.001        # Rand oben/unten abschneiden
fc = 0.13         # Rand links/rechts abschneiden

es = 5            # Erosion
_ds_black = 6
mpx = 1000
min_mask_object_size = 250        # minimale Objektgrösse
fv = 0            # Randfüllung schwarz
show_debug_images = False

sys.path.append('./')


def show_image(im):
    if not show_debug_images:
        return
    plt.figure(figsize=(8, 6))
    plt.imshow(im)
    plt.axis('off')
    plt.show()


def estimate_background_rgb(im, border_fraction=0.08):
    h, w = im.shape[:2]
    by = max(1, int(h * border_fraction))
    bx = max(1, int(w * border_fraction))
    border_pixels = np.concatenate([
        im[:by, :, :3].reshape(-1, 3),
        im[-by:, :, :3].reshape(-1, 3),
        im[:, :bx, :3].reshape(-1, 3),
        im[:, -bx:, :3].reshape(-1, 3),
    ])
    return np.median(border_pixels, axis=0)



def save_combined_region(fn, masked_image, regions, outputpath, min_num_pixels):
    relevant_regions = [region for region in regions if region.area >= min_num_pixels]
    if not relevant_regions:
        return 0

    minr = min(region.bbox[0] for region in relevant_regions)
    minc = min(region.bbox[1] for region in relevant_regions)
    maxr = max(region.bbox[2] for region in relevant_regions)
    maxc = max(region.bbox[3] for region in relevant_regions)

    crop = masked_image[minr:maxr, minc:maxc]
    output_name = f'{fn.parent.name}_{fn.stem}_object.jpg'
    outfile = Path(outputpath) / output_name
    imsave(outfile, crop.astype('uint8'))
    return len(relevant_regions)


def build_mask_for_background(im_hsv, bg_rgb, object_kind='wallball'):
    hue_channel = im_hsv[:, :, 0]
    saturation_channel = im_hsv[:, :, 1]
    value_channel = im_hsv[:, :, 2]

    bg_rgb_float = np.asarray(bg_rgb, dtype=np.float32) / 255.0
    bg_hsv = mpl_rgb_to_hsv(bg_rgb_float.reshape(1, 1, 3))[0, 0]
    bg_hue = bg_hsv[0] * 255
    bg_sat = bg_hsv[1] * 255
    bg_val = bg_hsv[2] * 255

    is_blue_background = bg_sat > 35 and 95 <= bg_hue <= 175

    if is_blue_background:
        # Dunkle Objektflächen müssen erhalten bleiben. Blaue Ballflächen
        # werden anschliessend über die geschlossene Objektkontur ergänzt.
        vth = int(bg_val * 0.75)
        ds = 3
        hue_distance = np.abs(((hue_channel - bg_hue + 127.5) % 255) - 127.5)
        background_like = (
            (hue_distance <= 12)
        )
        mask = (value_channel < vth) | ~background_like
        background_kind = 'blau/petrol'
    elif object_kind in {'kettlebell', 'dumbbell', 'dumbbells'}:
        # Farbe findet auch kleine Hanteln; die helleren, schwach
        # gesättigten Metallgriffe werden über den Value-Kanal ergänzt.
        vth = max(65, min(110, int(bg_val + 15)))
        bright_vth = max(125, min(185, int(bg_val + 65)))
        ds = 2
        mask = (
            ((value_channel > vth) & (saturation_channel > 30)) |
            ((value_channel > bright_vth) & (saturation_channel > 15))
        )
        background_kind = 'schwarz/grau'
    else:
        # Schwarze/graue Boeden sind oft texturiert und enthalten helle, aber
        # schwach gesaettigte Reflexe. Deshalb reicht V > vth nicht aus:
        # farbige Objektteile kommen ueber Saettigung rein, Metall/helle Teile
        # nur ueber eine deutlich hoehere Value-Schwelle.
        vth = max(85, min(125, int(bg_val + 22)))
        saturation_min = 38
        bright_vth = max(135, min(185, int(bg_val + 85)))
        ds = _ds_black
        mask = (
            ((value_channel > vth) & (saturation_channel > saturation_min)) |
            (value_channel > bright_vth)
        )
        background_kind = 'schwarz/grau'

    return mask, int(vth), int(ds), background_kind, bg_hue, bg_sat, bg_val


def process_image(fn, outputpath_Ausschnitte):
    print(f'Verarbeite: {fn}')
    object_kind = fn.parent.name.lower()
    is_weight = object_kind in {'kettlebell', 'dumbbell', 'dumbbells'}
    # Die kleinen Dumbbells haben bei 225 x 400 oft weniger als 1000 Pixel.
    minimum_area = 150 if is_weight else mpx
    minimum_fragment = 30 if is_weight else min_mask_object_size

    im = oe.load_file(fn)
    show_image(im)

    im_resized = (255 * resize(im, pr)).astype('uint8')
    print('Original:', im.shape, 'Verkleinert:', im_resized.shape)
    show_image(im_resized)

    im_filled = oe.fill_borders(
        im_resized,
        value_to_fill=fv,
        fraction_of_rows_to_remove=fr,
        fraction_of_cols_to_remove=fc
    )
    show_image(im_filled)

    im_float = im_resized.astype(np.float32) / 255.0
    im_hsv = mpl_rgb_to_hsv(im_float) * 255

    bg_rgb = estimate_background_rgb(im_resized)
    mask, vth, ds, background_kind, bg_hue, bg_sat, bg_val = build_mask_for_background(im_hsv, bg_rgb, object_kind)

    print(f'Hintergrund RGB geschaetzt: {bg_rgb.astype(int)}')
    print(f'Hintergrund HSV geschaetzt: H={bg_hue:.1f}, S={bg_sat:.1f}, V={bg_val:.1f}')
    print(f'Erkannt als Hintergrund: {background_kind}')
    print(f'Verwende Startwerte: vth={vth}, erosion_size={es}, dilation_size={ds}')

    show_image(mask)

    if is_weight:
        mask = oe.fill_borders(
            mask, value_to_fill=False,
            fraction_of_rows_to_remove=fr, fraction_of_cols_to_remove=fc
        )
    mask = remove_small_objects(mask.astype(bool), min_size=minimum_fragment)
    mask = remove_small_holes(mask, area_threshold=minimum_fragment)

    if is_weight:
        # Keine starke Erosion und keine konvexe Hülle: schmale Stangen
        # und die Öffnung im Kettlebell-Griff sollen erhalten bleiben.
        morphed_mask = binary_closing(mask, disk(2))
        morphed_mask = binary_opening(morphed_mask, disk(1))
        morphed_mask = remove_small_holes(morphed_mask, area_threshold=200)
        components = label(morphed_mask)
        candidates = [r for r in oe.regionprops(components) if r.area >= minimum_area]
        if candidates:
            # Pro Aufnahme liegt ein Gewicht im Bild; Bodenreflexe verwerfen.
            largest = max(candidates, key=lambda region: region.area)
            morphed_mask = components == largest.label
    elif background_kind == 'blau/petrol':
        # Kleine Störungen entfernen, Konturlücken schliessen und die
        # umschlossenen blauen Teile des Balls wieder aufnehmen.
        morphed_mask = binary_opening(mask, disk(2))
        morphed_mask = binary_closing(morphed_mask, disk(3))
        morphed_mask = oe.fill_borders(
            morphed_mask, value_to_fill=False,
            fraction_of_rows_to_remove=fr, fraction_of_cols_to_remove=fc
        )
        components = label(morphed_mask)
        candidates = oe.regionprops(components)
        if candidates:
            largest = max(candidates, key=lambda region: region.area)
            # Die Bälle sind konvex: so bleiben auch ähnlich blaue Flächen
            # erhalten, deren Kontur zur Hintergrundfarbe hin offen ist.
            morphed_mask = convex_hull_image(components == largest.label)
        morphed_mask = binary_fill_holes(morphed_mask)
    else:
        morphed_mask = oe.morphology_transform(
            mask, shape=square, erosion_size=es, dilation_size=ds
        )
    # Randbeschnitt erst auf die fertige Maske anwenden: aufgefüllte schwarze
    # Ränder dürfen bei dunklen Objekten nicht als Vordergrund erscheinen.
    morphed_mask = oe.fill_borders(
        morphed_mask, value_to_fill=False,
        fraction_of_rows_to_remove=fr, fraction_of_cols_to_remove=fc
    )
    morphed_mask = remove_small_objects(morphed_mask.astype(bool), min_size=minimum_fragment)
    show_image(morphed_mask)

    masked_image = oe.create_masked_image(im_filled, morphed_mask)
    show_image(masked_image)

    label_img = label(morphed_mask)
    show_image(label_img)

    regionlist, regions = oe.extract_regions(
        label_img,
        masked_image,
        fraction_of_rows_to_remove=fr
    )

    relevante_regionen = [reg for reg in regions if reg.area >= minimum_area]
    print(f'Gefundene Regionen: {len(regionlist)}, relevante Regionen: {len(relevante_regionen)}')

    return save_combined_region(
        fn,
        masked_image,
        regions,
        outputpath_Ausschnitte,
        minimum_area
    )


def main():
    outputpath_Ausschnitte = os.path.join(
        Ausgabepfad,
        'Ausschnitte'
    )

    oe.ask_path_creation(
        outputpath_Ausschnitte,
        yes_to_all=True
    )

    print(f'{len(imgfns)} Bilder werden verarbeitet.')

    total_regions = 0
    for fn in imgfns:
        total_regions += process_image(fn, outputpath_Ausschnitte)

    ausschnitte = list(Path(outputpath_Ausschnitte).glob('*'))

    print(f'Verarbeitung fertig. Relevante Regionen: {total_regions}')
    print(f'Extrahierte Ausschnitte im Ausgabeordner: {len(ausschnitte)}')

    for fn in ausschnitte[:10]:
        print(fn.name)


if __name__ == '__main__':
    main()
