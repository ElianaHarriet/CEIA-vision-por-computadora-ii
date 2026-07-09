# Project Notes — TP Visión por Computadora II

## Título

Sistema de Detección y Evaluación de Daños en Vehículos

## Propuesta (enviada)

**Problema real:** Cuando hay un choque, la evaluación del daño en los autos se hace de forma totalmente visual y manual por los peritos de los seguros, lo que lleva tiempo y puede ser muy subjetivo. Automatizar esto ayuda a acelerar los trámites y a tener un diagnóstico más rápido de la gravedad del golpe.

**Eje central:** Segmentación de Instancias para identificar el área exacta de la carrocería dañada (raspón/hundimiento).

**Capacidad adicional:** Estimación de Profundidad para medir qué tan hundido está el daño y clasificar automáticamente como leve, moderado o grave.

**Modelos tentativos:**
- Segmentación: YOLOv8-Seg (pre-entrenado en Roboflow)
- Profundidad: Depth-Anything o MiDaS

**Dataset:** https://universe.roboflow.com/college-qxdrt/car-damage-detection-ha5mm

## Respuesta del profesor

Recomienda comparar resultados de segmentación por instancias vs segmentación semántica.

**Dataset sugerido (semántica):** https://universe.roboflow.com/project-p5nyc/car-damages-v3gyz
