---
title: RAG Architectures — Patrones clave
date: 2024-11-15
tags: [rag, llm, arquitectura, retrieval]
source: personal
---

# RAG Architectures — Patrones clave

## Naive RAG
El patrón más básico: embed la query, busca top-K chunks, construye el prompt con esos chunks y llama al LLM. 

Problema principal: la calidad depende 100% de la calidad del retrieval. Si los chunks no son relevantes, el LLM alucina igual.

## Advanced RAG
Mejoras sobre naive:
- **Pre-retrieval**: reformulación de query (HyDE, query expansion)
- **Post-retrieval**: reranking con cross-encoder, compresión de contexto
- **Chunking inteligente**: chunking semántico vs fijo por tokens

## Modular RAG
Cada componente del pipeline es intercambiable. Puedes mezclar diferentes retrievers, rerankers y generadores.

## Cuándo usar cada uno
- Prototipo rápido → Naive RAG
- Producción con docs técnicos → Advanced RAG con reranking
- Casos complejos multi-step → Modular con routing
