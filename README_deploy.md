# Deploy en Railway — Guía paso a paso

## Qué necesitas
- Cuenta en [railway.app](https://railway.app) (gratis)
- Cuenta en [github.com](https://github.com) (gratis)
- Git instalado en Windows

---

## Paso 1 — Subir el código a GitHub

```powershell
# En la carpeta del proyecto
cd C:\Users\marha\Downloads\ai-notes-agent-v5b

# Inicializar repositorio git
git init
git add .
git commit -m "AI Notes Agent v1"

# Crear repo en github.com → New repository → "ai-notes-agent" (privado)
# Luego conectar:
git remote add origin https://github.com/TU_USUARIO/ai-notes-agent.git
git push -u origin main
```

---

## Paso 2 — Crear proyecto en Railway

1. Ve a [railway.app](https://railway.app) → **New Project**
2. Selecciona **Deploy from GitHub repo**
3. Autoriza Railway a acceder a tu GitHub
4. Selecciona el repo `ai-notes-agent`
5. Railway detecta automáticamente el `Procfile` y arranca el deploy

---

## Paso 3 — Configurar variables de entorno

En Railway → tu proyecto → **Variables** → añade:

| Variable | Valor |
|---|---|
| `VOYAGE_API_KEY` | tu key de Voyage AI |
| `ANTHROPIC_API_KEY` | tu key de Anthropic |
| `DATA_DIR` | `/data` |

---

## Paso 4 — Añadir volumen persistente (para la base de datos)

Sin volumen, el índice se borra en cada redeploy.

1. Railway → tu proyecto → **+ New** → **Volume**
2. Mount path: `/data`
3. Guarda → Railway reinicia automáticamente

Ahora `notes_index.db`, `voice_profile.json` y las notas se guardan en `/data` y sobreviven los redeploys.

---

## Paso 5 — Obtener tu URL pública

Railway → tu proyecto → **Settings** → **Domains** → **Generate Domain**

Te da una URL tipo: `https://ai-notes-agent-production.up.railway.app`

¡Ya tienes tu agente accesible desde cualquier dispositivo!

---

## Paso 6 — Indexar tus notas en la nube

Desde tu PC local, apunta el indexer a la URL de Railway:

```powershell
# Opción A: indexar localmente y subir el .db por SFTP/railway CLI
# Opción B (más simple): usar la interfaz web en la URL de Railway
#   → Noticias IA → Obtener noticias
#   → Indexar notas → pega texto directamente
```

O instala la Railway CLI:

```powershell
# Instalar Railway CLI
npm install -g @railway/cli

# Login
railway login

# Subir archivos al volumen
railway run python indexer.py index ./notes_sample
```

---

## Actualizaciones futuras

Cada vez que hagas cambios y los subas a GitHub, Railway redeploya automáticamente:

```powershell
git add .
git commit -m "mejora X"
git push
# Railway detecta el push y redeploya en ~60 segundos
```

---

## Costes estimados

| Plan | Precio | Incluye |
|---|---|---|
| Hobby (gratis) | $0/mes | 500h de compute, ideal para uso personal |
| Pro | $5/mes | Compute ilimitado + más memoria |

Para uso personal el plan gratuito es más que suficiente.
