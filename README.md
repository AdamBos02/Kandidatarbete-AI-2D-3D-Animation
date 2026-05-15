# AI-driven 2D-till-3D-animation med anatomiskt korrekta avatarer

Kandidatarbete vid Chalmers Tekniska Högskola, Institutionen för Fysik, 2026.

## Författare
- Agnes Bengtsson
- Adam Boström
- Victor Jakobsson
- Erik Tjeder

## Beskrivning
Detta projekt utvärderar AI-baserade pose-estimeringsmetoder för att 
omvandla 2D-video till 3D-animationer av friidrottsrörelser. Egenutvecklade 
arbetsflöden baserade på öppen källkod jämförs med kommersiella lösningar 
mot referensdata från ett Qualisys rörelsefångstsystem.

## Modeller och arbetsflöden
Följande arbetsflöden utvärderades:

| 2D-modell | 3D-modell | Format |
|-----------|-----------|--------|
| MMPose (RTMw-x) | MotionBERT | COCO WholeBody → HALPE-26 → Human3.6M |
| MMPose (RTMw-x) | MotionAGFormer | COCO WholeBody → Human3.6M |
| ViTPose-Large | MotionBERT | COCO-17 → HALPE-26 → Human3.6M |
| ViTPose-Large | MotionAGFormer | COCO-17 → Human3.6M |
| MediaPipe | MediaPipe | BlazePose |

Kommersiella system som utvärderades: QuickMagic och Mimem.

## Datainsamling
Referensdata samlades in vid Chalmers FUSE Fysiolabb med hjälp av:
- Qualisys Track Manager 2026.1
- 12 Miqus M3-kameror
- Inspelningsfrekvens: 120 fps

Videomaterial spelades in med två iPhone 13 i 1080p och 60 fps.

Övningar som analyserades:
- Jämfotahopp
- Hoppsteg med knälyft

## Kod
I detta GitHub-repo finns all kod som har möjliggjort vårt arbete tillgänglig,
samt de dependencies och conda-miljöer som har krävts för att använda de 
externa modellerna.

## Rapport
____

## Externa modeller
Dessa modeller har legat till grund för det arbete som har utförts i detta kandidatarbete
- [MMPose](https://github.com/open-mmlab/mmpose)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer) + checkpoint från ([AthletePose3D](https://github.com/calvinyeungck/AthletePose3D))
