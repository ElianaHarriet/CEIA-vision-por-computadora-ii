# CEIA - Visión por Computadora II

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Repositorio para el curso de Visión por Computadora II de la Carrera de Especialización en Inteligencia Artificial (CEIA) - FIUBA.

## Dataset

Car Damages forkeado en Roboflow. 4 clases:

| ID | Clase |
|----|-------|
| 0 | Minor Damage (Dent) |
| 1 | Minor Damage (Scratch) |
| 2 | No Damage |
| 3 | Severe Damage |

Descargar y preparar:
```bash
cp .env.template .env   # completar ROBOFLOW_API_KEY
poetry run python car_damage_detection/src/utils/download_dataset.py
```

Genera dos formatos en `car_damage_detection/data/car-damages-ready/`:
- `instance/` — etiquetas YOLOv8 .txt para segmentación de instancias
- `semantic/` — máscaras PNG para segmentación semántica

## Dependencias

```bash
pip install poetry
poetry install
```

## Integrantes

| Nombre | Email |
|---|---|
| Santiago Bartolini Rizzo | santiagobartolini@gmail.com |
| Luis Ali | aliluis@gmail.com |
| Eliana Harriet | eharriet@fi.uba.ar |

## Submódulo

```
vision_computadora_II/  (rama: VpC2_2026)
└─ https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/vision_computadora_II
```

```bash
git submodule update --init --recursive
```

## Licencia

Este proyecto está licenciado bajo Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
