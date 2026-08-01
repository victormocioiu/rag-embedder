"""Compare the deployed endpoint against the PyTorch reference.

Ten minutes, and it is what stands between you and publishing numbers for a
subtly broken model. Especially relevant on AVX-512-without-VNNI hardware, where
U8S8 saturation can silently degrade int8 accuracy.

    uv sync --group parity
    uv run python scripts/parity_check.py --url https://rag-embedder.<tailnet>.ts.net

Expect (per-channel int8, measured):
    p5   >= 0.99    below this, suspect quantization saturation
                    -- rebuild with --skip-quantize to isolate
                    (an fp32 export should sit >= 0.9999 on every pair)
    mean >= 0.994

The gate is p5 + mean rather than min: the sentence set deliberately includes
inputs far past the 512-token truncation boundary and degenerate strings,
where quantization error is largest (~0.98 there is expected).

Gate against the DEPLOYED endpoint, not a local run: dynamically quantized
inference is ISA-dependent (identical model bytes score differently on
different CPUs), and the serving CPU is the number that matters. Measured on
the AMD EPYC Genoa embed node: p5 0.9941, mean 0.9961.
"""

import argparse
import math

import httpx
import numpy as np

# ~200 sentences: varied languages, lengths, and domains. The point is token
# diversity (scripts, digits, punctuation, length buckets), not semantics.
BASE = [
    # English -- prose, support, commerce, technical
    "The quick brown fox jumps over the lazy dog.",
    "Refund requests must be submitted within 30 days of purchase.",
    "SKU AB-1234-XL is discontinued as of Q3.",
    "Please reset your password using the link we emailed you.",
    "The invoice total of $4,321.09 does not match the purchase order.",
    "Our office is closed on public holidays and weekends.",
    "The firmware update fixes a race condition in the Bluetooth stack.",
    "Yeah, that works for me, see you at ten.",
    "Delivery windows are estimates, not guarantees, per section 12 of the terms.",
    "The patient reported intermittent chest pain radiating to the left arm.",
    "Mix two cups of flour with a pinch of salt before adding the yeast.",
    "Quarterly revenue grew 14% year over year, driven by subscriptions.",
    "The hiking trail is closed above 2,000 meters due to early snowfall.",
    "Error 429: too many requests, retry after sixty seconds.",
    "The museum's impressionist wing reopens to the public next spring.",
    "Insufficient permissions to modify this resource.",
    "A single misconfigured DNS record took the whole checkout flow down.",
    "OK.",
    "No refunds on digital goods.",
    "The committee postponed the vote until further notice.",
    # French
    "Le renard brun rapide saute par-dessus le chien paresseux.",
    "Veuillez conserver votre reçu comme preuve d'achat.",
    "La réunion est reportée à jeudi prochain à quatorze heures.",
    "Les résultats trimestriels dépassent les attentes des analystes.",
    "Ce médicament doit être pris au cours des repas.",
    "La gare est fermée pour travaux jusqu'en septembre.",
    # German
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Bitte bewahren Sie den Beleg für eventuelle Rückfragen auf.",
    "Die Lieferung verzögert sich wegen eines Streiks um drei Werktage.",
    "Das Software-Update behebt mehrere sicherheitskritische Fehler.",
    "Der Mietvertrag verlängert sich automatisch um zwölf Monate.",
    # Spanish
    "El zorro marrón rápido salta sobre el perro perezoso.",
    "Su pedido ha sido enviado y llegará en dos días hábiles.",
    "La factura vence el quince de marzo.",
    "El museo ofrece entrada gratuita los domingos por la mañana.",
    # Romanian
    "Vulpea maro sare rapid peste câinele leneș.",
    "Factura trebuie achitată în termen de treizeci de zile.",
    "Trenul de București are o întârziere de patruzeci de minute.",
    "Vremea se răcește accentuat începând de mâine seară.",
    # Italian / Portuguese / Dutch
    "La consegna è prevista entro la fine della settimana.",
    "Il contratto scade il trentuno dicembre.",
    "O pagamento foi recusado pelo banco emissor.",
    "A reunião foi adiada para a próxima segunda-feira.",
    "De trein naar Amsterdam vertrekt van spoor vijf.",
    "Uw pakket ligt klaar bij het afhaalpunt.",
    # Polish / Russian / Ukrainian
    "Zamówienie zostanie wysłane w ciągu dwóch dni roboczych.",
    "Prosimy o kontakt z działem obsługi klienta.",
    "Быстрая коричневая лиса перепрыгивает через ленивую собаку.",
    "Оплата не прошла, попробуйте другую карту.",
    "Ваше замовлення буде доставлено протягом трьох днів.",
    # Chinese / Japanese / Korean
    "敏捷的棕色狐狸跳过了懒惰的狗。",
    "您的订单已发货，预计三天内送达。",
    "会议改到下周四下午两点举行。",
    "素早い茶色の狐がのろまな犬を飛び越える。",
    "ご注文の商品は明日発送予定です。",
    "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
    "주문하신 상품이 오늘 발송되었습니다.",
    # Arabic / Hebrew / Hindi / Turkish
    "الثعلب البني السريع يقفز فوق الكلب الكسول.",
    "سيتم شحن طلبك خلال يومي عمل.",
    "השועל החום המהיר קופץ מעל הכלב העצלן.",
    "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूद जाती है।",
    "आपका ऑर्डर तीन दिनों में डिलीवर होगा।",
    "Hızlı kahverengi tilki tembel köpeğin üzerinden atlar.",
    "Siparişiniz iki iş günü içinde kargoya verilecektir.",
    # Nordic / Finnish / Hungarian / Greek
    "Den snabba bruna räven hoppar över den lata hunden.",
    "Nopea ruskea kettu hyppää laiskan koiran yli.",
    "A gyors barna róka átugorja a lusta kutyát.",
    "Η γρήγορη καφέ αλεπού πηδάει πάνω από τον τεμπέλη σκύλο.",
    # Vietnamese / Thai / Indonesian
    "Con cáo nâu nhanh nhẹn nhảy qua con chó lười biếng.",
    "สุนัขจิ้งจอกสีน้ำตาลกระโดดข้ามสุนัขขี้เกียจ",
    "Rubah cokelat yang cepat melompati anjing yang malas.",
    # Mixed scripts, numbers, code-ish, URLs, edge cases
    "Order #98213-B shipped via DHL, tracking JD014600003RO.",
    "π is approximately 3.14159, e is approximately 2.71828.",
    "SELECT count(*) FROM orders WHERE status = 'pending';",
    "The endpoint POST /v2/embeddings accepts up to 96 texts per call.",
    "Visit https://docs.example.com/quickstart for setup instructions.",
    "CPU: 2 cores, RAM: 4 GB, disk: 80 GB NVMe.",
    "N/A",
    "12345",
    "café, naïve, façade, jalapeño, über, señor",
    "The temperature dropped from 21.5°C to -3°C overnight.",
    "«质量第一» was printed on every box in the warehouse.",
    "Mon–Fri 09:00–17:30 CET, excluding bank holidays.",
    "v2.7.1-rc3 introduces breaking changes to the auth middleware.",
    "The mitochondria is the powerhouse of the cell.",
    "All your base are belong to us.",
    "E = mc² remains the most famous equation in physics.",
]

LONG_PARAGRAPH = (
    "The migration ran for six hours before anyone noticed that the replica "
    "was silently dropping writes; by then the backlog had grown past two "
    "million events, the on-call engineer had been paged twice, and the only "
    "safe path forward was a full resync from the primary, which meant "
    "another night of degraded latency for every customer in the region. "
)


def build_sentences() -> list[str]:
    sentences = list(BASE)
    # medium: cross-language concatenations, deterministic
    n = len(BASE)
    i = 0
    while len(sentences) < 196:
        a, b = BASE[i % n], BASE[(i + 7) % n]
        sentences.append(f"{a} {b}")
        i += 1
    # long: near and past the 512-token truncation boundary
    sentences.append(LONG_PARAGRAPH)
    sentences.append(LONG_PARAGRAPH * 3)
    sentences.append(LONG_PARAGRAPH * 8)
    sentences.append(" ".join(BASE[:40]))
    return sentences


def embed_remote(url: str, texts: list[str], input_type: str, batch: int = 32) -> np.ndarray:
    out: list[list[float]] = []
    with httpx.Client(timeout=60.0) as client:
        for i in range(0, len(texts), batch):
            r = client.post(f"{url.rstrip('/')}/embed",
                            json={"texts": texts[i:i + batch], "input_type": input_type})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
            print(f"  remote {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return np.asarray(out, dtype=np.float64)


def embed_reference(model_name: str, texts: list[str], input_type: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    prefixed = [f"{input_type}: {t}" for t in texts]
    return np.asarray(
        model.encode(prefixed, normalize_embeddings=True, batch_size=32,
                     show_progress_bar=False),
        dtype=np.float64,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="https://rag-embedder.<tailnet>.ts.net")
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--input-type", default="passage", choices=["query", "passage"])
    p.add_argument("--threshold", type=float, default=0.99)
    p.add_argument("--json", default=None,
                   help="also write per-pair cosines + text lengths to this file")
    a = p.parse_args()

    texts = build_sentences()
    print(f"{len(texts)} sentences")

    print("embedding via endpoint (int8)...")
    remote = embed_remote(a.url, texts, a.input_type)
    print("embedding via sentence-transformers (PyTorch fp32)...")
    ref = embed_reference(a.model, texts, a.input_type)

    norms = np.linalg.norm(remote, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print(f"FAIL: endpoint vectors not unit-norm (min {norms.min():.6f}, "
              f"max {norms.max():.6f})")
        return 1

    cos = np.sum(remote * ref, axis=1)  # both unit-norm -> dot == cosine
    order = np.argsort(cos)

    print(f"\nint8 endpoint vs PyTorch reference, n={len(texts)}")
    print(f"  min    {cos.min():.6f}")
    print(f"  p5     {np.percentile(cos, 5):.6f}")
    print(f"  mean   {cos.mean():.6f}")
    print(f"  max    {cos.max():.6f}")
    print("\nworst 5:")
    for idx in order[:5]:
        print(f"  {cos[idx]:.6f}  {texts[idx][:70]!r}")

    p5 = float(np.percentile(cos, 5))
    mean = float(cos.mean())
    if a.json:
        import json

        with open(a.json, "w") as f:
            json.dump({"parity": {
                "cos": [round(float(c), 6) for c in cos],
                "char_len": [len(t) for t in texts],
                "threshold": a.threshold,
            }}, f, indent=1)
        print(f"wrote {a.json}")
    if p5 < a.threshold or mean < 0.994:
        below = int((cos < a.threshold).sum())
        print(f"\nFAIL: p5 {p5:.6f} (need >= {a.threshold}), mean {mean:.6f} "
              f"(need >= 0.994); {below}/{len(texts)} pairs below {a.threshold}")
        print("suspect quantization saturation -- rebuild with --skip-quantize "
              "and compare fp32 to isolate export vs quantization")
        return 1
    print(f"\nPASS: p5 {p5:.6f}, mean {mean:.6f}, "
          f"floor {math.floor(cos.min() * 1e4) / 1e4} (n={len(texts)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
