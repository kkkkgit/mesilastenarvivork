# Mesilaste närvivõrk ja objektituvastus

TalTech Tartu Kolledži praktikumitöö — mesilaste klassifitseerimine konvolutsioonilise närvivõrguga (CNN) ja reaalajas objektituvastus kaameraga.

## Projekti sisu

| Fail | Kirjeldus |
|------|-----------|
| `mesilaste_narvivork.py` | Mesilaste CNN (BeeNet) treenimine, evalueerimine ja andmete augmentatsioon |
| `kaamera.py` | MacBooki kaamera + treenitud BeeNet mudel — pildista ja klassifitseeri |
| `inimeste_segmenteerimine.py` | YOLO + SAM2 — inimeste tuvastus ja pikslitäpne segmenteerimine kaameraga |
| `sam2_kaamera.py` | SAM2 interaktiivne segmenteerimine — kliki objektile ja lõika välja |
| `yolo_kaamera.py` | YOLOv8 reaalajas objektituvastus kaameraga (80 objektiklassi) |

## Eeldused

- Python 3.10+
- macOS (Apple Silicon M1/M2/M3 MPS tugi) või Linux/WSL
- Veebiühendus (andmestiku ja mudelite allalaadimiseks)

## Paigaldamine

```bash
# Loo ja aktiveeri virtuaalkeskkond
python3 -m venv mesilased
source mesilased/bin/activate

# Paigalda sõltuvused
pip install torch torchvision matplotlib requests Pillow opencv-python ultralytics
```

## Kasutamine

### 1. Mesilaste närvivõrgu treenimine

```bash
python3 mesilaste_narvivork.py
```

Laeb alla mesilaste andmestiku, treenib BeeNet mudeli 2×20 epohhi (tavaline + augmentatsiooniga) ja salvestab `model.pkl` faili.

### 2. Mesilaste klassifitseerimine kaameraga

```bash
python3 kaamera.py
```

Laeb treenitud mudeli ja avab kaamera. `SPACE` pildistab ja klassifitseerib, `Q` väljub.
Mudelit saab vahetada vastavalt vajadusele. Kas kasutad mesilaste varianti või Yolo oma.

Reaalajas objektituvastus bounding box'idega. Tuvastab 80 erinevat objekti (inimesed, autod, loomad jne). `Q` väljub.

### 4. Inimeste segmenteerimine (YOLO + SAM2)

```bash
python3 inimeste_segmenteerimine.py
```

`SPACE` tuvastab inimesed ja lõikab nad pikslitäpselt välja. Näitab kahte vaadet kõrvuti: overlay + eraldatud inimesed. Toetab ka tausta hägustamist. `C` puhastab, `Q` väljub.

### 5. SAM2 interaktiivne segmenteerimine

```bash
python3 sam2_kaamera.py
```

Kliki hiirega mis tahes objektile ja SAM2 lõikab selle välja:
- **Vasak klikk** — lisa punkt (segmenteeri see objekt)
- **Parem klikk** — negatiivne punkt (välista ala)
- **SPACE** — jooksuta segmenteerimine
- **C** — puhasta punktid
- **Q** — välju

## Tehnoloogiad

- **PyTorch** — närvivõrgu treenimine ja inferents
- **torchvision** — pilditöötlus ja andmete augmentatsioon
- **YOLOv8** (Ultralytics) — reaalajas objektituvastus
- **SAM2** (Segment Anything Model 2) — pikslitäpne segmenteerimine
- **OpenCV** — kaamera ja pilditöötlus
- **MPS** (Metal Performance Shaders) — Apple Silicon GPU kiirendus

## .gitignore

Reposse ei lükatud:
- `mesilased/` — virtuaalkeskkond
- `bee3.zip`, `bee_new/` — andmestik (laetakse automaatselt)
- `model.pkl`, `*.pt` — mudelifailid (genereeritakse/laetakse automaatselt)