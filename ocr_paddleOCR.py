#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, asdict
from typing import List, Tuple, Any, Dict

import numpy as np
import cv2
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from PIL import Image
from tqdm import tqdm


@dataclass
class OCRBox:
    polygon: List[List[float]]  # 4 points [[x1,y1],...,[x4,y4]] dans l'ordre
    text: str
    confidence: float


@dataclass
class PageResult:
    page_index: int          # 0-based
    width: int               # pixels après rasterisation
    height: int
    dpi: int
    boxes: List[OCRBox]


@dataclass
class DocumentResult:
    file_path: str
    pages: List[PageResult]
    avg_confidence: float
    total_chars: int


def improve_readability(img_bgr: np.ndarray) -> np.ndarray:
    """
    Prétraitement léger mais robuste :
    - passage en niveaux de gris
    - égalisation adaptative du contraste (CLAHE)
    - débruitage léger
    - sharpen
    - tentative de deskew (si détectable)
    """
    # Gris
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Débruitage
    gray = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)

    # Sharpen
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # Binarisation douce pour deskew
    thr = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Deskew (Hough-based). Si pas de lignes significatives, on garde tel quel.
    angle = estimate_skew_angle(thr)
    if angle is not None and abs(angle) > 0.5 and abs(angle) < 15:
        sharp = rotate_image(sharp, angle)

    # Retour en 3 canaux (Paddle accepte np.ndarray HxW ou HxWxC ; on choisit 3 canaux)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def estimate_skew_angle(bin_img: np.ndarray) -> float | None:
    # Cherche des lignes pour estimer une inclinaison globale
    edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=180)
    if lines is None:
        return None

    angles = []
    for rho_theta in lines[:200]:  # limiter
        rho, theta = rho_theta[0]
        # Convertir l'angle radian en degrés autour de l'horizontal
        deg = (theta * 180.0 / np.pi)
        # convertir en une inclinaison autour de 0 (par rapport à l'horizontale)
        # 0 ~ horizontal, 90 ~ vertical ; on veut l'écart à 90
        if deg > 90:
            deg -= 180
        if -60 < deg < 60:
            angles.append(deg)

    if not angles:
        return None

    # médiane robuste
    return float(np.median(angles))


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def run_paddle_ocr(
    images_bgr: List[np.ndarray],
    lang: str = "fr",
    use_gpu: bool = False,
    rec: bool = True,
    cls: bool = True
) -> List[PageResult]:
    # Instancier PaddleOCR (une fois) — modèle FR par défaut, sinon multi-lang
    ocr = PaddleOCR(
        use_angle_cls=cls,
        lang=lang,
        use_gpu=use_gpu,
        show_log=False
    )

    page_results: List[PageResult] = []
    for idx, img_bgr in enumerate(tqdm(images_bgr, desc="OCR", unit="page")):
        # PaddleOCR attend BGR ou chemin ; nous passons le np.ndarray
        result = ocr.ocr(img_bgr, cls=cls)
        # result est une liste par image ; nous traitons l'unique page -> result[0]
        detrecs = result[0] if result and len(result) > 0 else []

        boxes = []
        for det in detrecs:
            # det: [ [[x1,y1],...,[x4,y4]] , (text, score) ]
            polygon = det[0]
            text = det[1][0]
            conf = float(det[1][1]) if det[1][1] is not None else 0.0
            boxes.append(OCRBox(polygon=polygon, text=text, confidence=conf))

        h, w = img_bgr.shape[:2]
        page_results.append(PageResult(page_index=idx, width=w, height=h, dpi=300, boxes=boxes))

    return page_results


def pages_to_images(pdf_path: str, dpi: int = 300, poppler_path: str | None = None) -> List[np.ndarray]:
    pil_pages: List[Image.Image] = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    images = []
    for p in pil_pages:
        # Convert PIL -> OpenCV BGR
        img = cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
        img = improve_readability(img)
        images.append(img)
    return images


def aggregate_stats(pages: List[PageResult]) -> Tuple[float, int]:
    all_confs = []
    total_chars = 0
    for p in pages:
        for b in p.boxes:
            all_confs.append(b.confidence)
            total_chars += len(b.text or "")
    avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
    return avg_conf, total_chars


def save_outputs(doc: DocumentResult, out_dir: str, base_name: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{base_name}.ocr.json")
    txt_path = os.path.join(out_dir, f"{base_name}.txt")

    # JSON structuré
    with open(json_path, "w", encoding="utf-8") as f:
        payload = asdict(doc)
        # convertir dataclasses imbriquées
        payload["pages"] = [
            {
                "page_index": p.page_index,
                "width": p.width,
                "height": p.height,
                "dpi": p.dpi,
                "boxes": [asdict(b) for b in p.boxes],
            }
            for p in doc.pages
        ]
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # TXT concaténé (ordre de détection)
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in doc.pages:
            f.write(f"===== Page {p.page_index + 1} =====\n")
            for b in p.boxes:
                f.write(b.text.strip() + "\n")
            f.write("\n")

    return json_path, txt_path


def main():
    parser = argparse.ArgumentParser(
        description="OCÉRISATION de factures PDF (texte + positions) avec PaddleOCR."
    )
    parser.add_argument("pdf", help="Chemin du PDF à traiter.")
    parser.add_argument(
        "-o", "--out-dir", default="ocr_outputs",
        help="Dossier de sortie (JSON + TXT)."
    )
    parser.add_argument(
        "--lang", default="fr",
        help="Langue du modèle PaddleOCR (ex: fr, en, es, de, it, fr+en: utilisez 'multilang' si besoin)."
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Utiliser le GPU si disponible pour PaddleOCR."
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI pour la rasterisation des pages PDF."
    )
    parser.add_argument(
        "--poppler-path", default=None,
        help="Chemin du dossier Poppler (Windows). Laissez vide si Poppler est dans le PATH."
    )
    parser.add_argument(
        "--min-conf", type=float, default=0.55,
        help="Confiance moyenne minimale requise pour considérer l'OCR comme valide."
    )
    parser.add_argument(
        "--min-chars", type=int, default=30,
        help="Nombre minimal de caractères totaux requis pour considérer l'OCR comme valide."
    )
    args = parser.parse_args()

    pdf_path = args.pdf
    if not os.path.isfile(pdf_path):
        print(f"[ERREUR] Fichier introuvable: {pdf_path}", file=sys.stderr)
        sys.exit(2)

    try:
        images = pages_to_images(pdf_path, dpi=args.dpi, poppler_path=args.poppler_path)
        if not images:
            print("[ERREUR] Aucune page n'a pu être rasterisée depuis le PDF.", file=sys.stderr)
            sys.exit(2)

        page_results = run_paddle_ocr(images, lang=args.lang, use_gpu=args.gpu)
        avg_conf, total_chars = aggregate_stats(page_results)

        doc = DocumentResult(
            file_path=os.path.abspath(pdf_path),
            pages=page_results,
            avg_confidence=avg_conf,
            total_chars=total_chars
        )

        # Critères de robustesse / arrêt si OCR insuffisant
        if total_chars < args.min_chars or avg_conf < args.min_conf:
            msg = (
                f"[ECHEC OCR] Texte trop faible ou peu fiable. "
                f"total_chars={total_chars} (min {args.min_chars}), "
                f"avg_conf={avg_conf:.3f} (min {args.min_conf}). "
                f"Traitement stoppé."
            )
            print(msg, file=sys.stderr)
            # On exporte quand même un JSON minimal pour inspection
            base = os.path.splitext(os.path.basename(pdf_path))[0]
            save_outputs(doc, args.out_dir, base)
            sys.exit(3)

        # Sauvegarde des résultats
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        json_path, txt_path = save_outputs(doc, args.out_dir, base)

        print("[SUCCÈS] OCR terminé.")
        print(f" - Confiance moyenne : {avg_conf:.3f}")
        print(f" - Caractères totaux : {total_chars}")
        print(f" - JSON positions   : {json_path}")
        print(f" - Texte concaténé  : {txt_path}")

    except Exception as e:
        print(f"[ERREUR] Échec du traitement : {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
