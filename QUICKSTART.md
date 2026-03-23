# Guide de Démarrage Rapide - RAG-Anything

## Démarrage en 5 Minutes

### 1. Installation d'Ollama et des Modèles

```bash
# Télécharger Ollama depuis https://ollama.ai
# Ou sous Windows via winget
winget install Ollama.Ollama

# Télécharger les trois modèles requis
ollama pull llama3.1:8b        # génération de texte
ollama pull llava:7b           # analyse d'images (vision)
ollama pull nomic-embed-text   # embeddings (768 dimensions)
```

### 2. Installation des Dépendances Python

```bash
# Créer et activer l'environnement virtuel
python -m venv anything
# Windows
.\anything\Scripts\activate
# Linux/macOS
source anything/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Démarrer l'API

```bash
python api.py
```

L'API sera disponible sur **http://localhost:8000** — Swagger UI sur **http://localhost:8000/docs**

### 4. Ingérer les Documents

Placez vos PDFs dans le dossier `./donnees rag/`, puis lancez l'ingestion :

```bash
curl -X POST http://localhost:8000/ingest/folder
```

> **Temps estimé** : 2–5 minutes selon le nombre de documents. Les données indexées sont persistées dans `data/rag_anything_storage/`.

### 5. Tester l'API

#### Health Check
```
GET http://localhost:8000/health
```

#### Poser une Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Quelles sont les règles de TVA?", "mode": "hybrid"}'
```

**Modes de requête disponibles :**
| Mode | Description |
|------|-------------|
| `hybrid` | Local + global (recommandé) |
| `naive` | Vectoriel pur |
| `local` | Entités proches dans le graphe |
| `global` | Thèmes transversaux |

#### Avec Postman

1. Ouvrir Postman → **Import**
2. Sélectionner `postman_collection.json`
3. Lancer les requêtes préconfigurées

---

## Exemples de Questions

- "Quelles sont les obligations de facturation ?"
- "Que dit l'article 289 du CGI ?"
- "Comment déclarer la TVA ?"
- "Quelles sont les sanctions en cas de non-respect ?"

---

## Résolution de Problèmes Courants

### "Impossible de se connecter à Ollama"
```bash
# Vérifier qu'Ollama tourne et que les modèles sont présents
ollama list

# Télécharger les modèles manquants
ollama pull llama3.1:8b
ollama pull llava:7b
ollama pull nomic-embed-text
```

### "Vector store vide" / Réponses de mauvaise qualité
```bash
# Vérifier que des PDFs sont présents dans ./donnees rag/
# puis relancer l'ingestion
curl -X POST http://localhost:8000/ingest/folder
```

### Vider le cache LLM (debug)
Supprimer ou vider le fichier :
```
data/rag_anything_storage/kv_store_llm_response_cache.json
```

### "Module introuvable"
```bash
pip install -r requirements.txt --force-reinstall
```

---

## Vérifier les Résultats

```
GET http://localhost:8000/stats
GET http://localhost:8000/docs    # Documentation interactive
```

Logs disponibles dans `logs/`.

---

## Checklist de Démarrage

- [ ] Ollama installé et en cours d'exécution
- [ ] Modèles `llama3.1:8b`, `llava:7b`, `nomic-embed-text` téléchargés
- [ ] Environnement virtuel Python créé et activé
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] PDFs placés dans `./donnees rag/`
- [ ] API démarrée (`python api.py`)
- [ ] Ingestion lancée (`POST /ingest/folder`)
- [ ] Health check OK (`GET /health`)
- [ ] Test de requête réussi (`POST /query`)

---

## Configuration

Tous les paramètres sont dans `config.yaml` (modèles, chemins, mode de requête par défaut). Voir la section **Configuration** du `CLAUDE.md` pour les détails.
