---
title: Prompt Engineering — Técnicas avanzadas
date: 2024-12-03
tags: [prompting, llm, claude, gpt]
source: personal
---

# Prompt Engineering — Técnicas avanzadas

## Chain of Thought (CoT)
Pedir al modelo que "piense paso a paso" mejora drásticamente el razonamiento en tareas complejas. No funciona igual en modelos pequeños.

## Few-shot vs Zero-shot
Few-shot da ejemplos del formato esperado. Muy útil para outputs estructurados (JSON, tablas). Zero-shot funciona mejor cuando el modelo ya conoce bien la tarea.

## System prompt design
El system prompt es el contrato con el modelo. Debe incluir:
1. Rol y personalidad
2. Restricciones claras
3. Formato de output esperado
4. Ejemplos si el formato es complejo

## Técnica de "persona experta"
En vez de decir "responde esta pregunta", decir "eres un experto en X con 20 años de experiencia, responde como tal". Mejora la especificidad y reduce el hedging excesivo.

## Errores comunes
- Prompts demasiado largos sin estructura → el modelo pierde el hilo
- Instrucciones contradictorias → comportamiento impredecible
- No especificar el formato de output → respuestas inconsistentes
