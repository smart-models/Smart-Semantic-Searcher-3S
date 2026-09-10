# Ingest via MCP: pattern operativo (costo zero)

## TL;DR per lo sviluppatore

Il server MCP **non** espone (e non dovrebbe esporre) operazioni filesystem
sulla directory `data/` del middleware. Esistono due vie MCP-native per
inserire contenuti, più una terza via operativa per file di grandi
dimensioni. Questo documento spiega quando usare l'una o l'altra, perché non
abbiamo aggiunto un CRUD file ai tool MCP, e quale passo shell è richiesto
nel caso "file grosso".

---

## Il gap che si è manifestato in produzione

Un agente MCP-only che ha già pronto un file JSON di chunk (es. generato da
uno script di preprocessing) non può scriverlo in `data/` usando i tool MCP.
`qdrant_ingest_file` legge solo file già presenti sotto
`middleware-qdrant-manager/data/`, e il server MCP non ha alcun tool per
stage-are file lì. Risultato: l'agente deve "uscire" dal MCP e usare la
shell dell'host per copiare il file, poi tornare in MCP per l'ingest.

**Questo è intenzionale e va mantenuto.** Il resto del documento spiega il
perché e il pattern corretto.

---

## Le tre vie di ingest

### Via 1 — `qdrant_ingest_chunks` (MCP-native, inline)

Per contenuti di dimensioni contenute già disponibili in memoria/nel
contesto dell'agente.

```json
qdrant_ingest_chunks {
  "source_id": "manual-v1",
  "collection_name": "docs",
  "chunks": [
    {"id": 1, "text": "Qdrant stores vectors.", "lang": "en"},
    {"id": 2, "text": "The middleware embeds text before writing.", "lang": "en"}
  ]
}
```

- Nessun file su disco, nessun passo shell.
- `source_id` è obbligatorio (nessun nome file da cui dedurlo).
- Campi costanti su tutti i chunk possono andare in `metadata` (es.
  `{"author": "...", "work": "..."}`) per ridurre le dimensioni del payload.
- **Limite reale**: 4 MiB per richiesta, imposto dal `max_request_body_size`
  del transport streamable-HTTP del server MCP, *prima* che il tool venga
  eseguito — l'agente riceve un rifiuto a livello di trasporto, senza alcun
  messaggio tool-level che lo spieghi (il branch 413 in `errors.py` non
  viene mai raggiunto per questo caso). **Non** è il body-size del
  middleware: `REQUEST_BODY_MAX_BYTES` vale 26_214_400 byte (25 MiB) di
  default, una soglia che passando dal MCP non si raggiunge mai.
- **Limite pratico, più basso del precedente**: i chunk inline attraversano
  la context window dell'agente. Un corpus da ~1.3 MB starebbe dentro i
  4 MiB, ma inlinearlo è impraticabile molto prima di essere troppo grande.
  A quella scala usare la Via 3, o spezzare su più chiamate
  `qdrant_ingest_chunks` che condividono lo stesso `source_id`.

### Via 2 — `qdrant_ingest_file` (MCP-native, file già staged)

Per file già presenti sotto `data/` del middleware (es. caricati dal
deploy, da un pipeline CI, o staged manualmente dall'operatore).

```json
qdrant_ingest_file {
  "file_path": "corpus/manual.json",
  "collection_name": "docs"
}
```

- `file_path` è relativo a `middleware-qdrant-manager/data/`.
- Path assoluti e segmenti `..` sono rifiutati con HTTP 400 (sandbox).
- `source_id` di default = nome file senza estensione.
- Il file deve già esistere: il tool MCP **non** lo crea.

### Via 3 — File grosso + passo shell esplicito (operativa)

Per file di grandi dimensioni generati fuori dal middleware. È il pattern
da documentare come "procedura operativa" negli agent-facing runbook.

**Step 1 — prepara il file** (lato host, fuori dal MCP):
il file JSON deve avere un array `chunks` al top-level:
```json
{ "chunks": [ {"id": 0, "text": "...", ...}, ... ] }
```

**Step 2 — copia in `data/`** (shell dell'host, NON un tool MCP):
```powershell
$dst = "<repo>\middleware-qdrant-manager\data\corpus"
New-Item -ItemType Directory -Path $dst -Force | Out-Null
Copy-Item "<percorso sorgente>\file.json" -Destination "$dst\file.json" -Force
```
Su Linux/macOS:
```bash
mkdir -p "<repo>/middleware-qdrant-manager/data/corpus"
cp "<percorso sorgente>/file.json" "<repo>/middleware-qdrant-manager/data/corpus/file.json"
```

**Step 3 — ingest via MCP**:
```json
qdrant_ingest_file {
  "file_path": "corpus/file.json",
  "collection_name": "docs",
  "source_id": "file",
  "batch_size": 8
}
```

Il passo shell è **esplicito e visibile**: non è nascosto dietro un tool
MCP, quindi l'operatore sa di aver scritto su `data/` e resta responsabile
di cosa ci finisce.

---

## Esempio reale seguito in questo progetto

Ingest di *I Promessi Sposi* (39 capitoli, ~220k parole, ~1.3 MB JSON):

1. Script Python `build_promessi_sposi_chunks.py` genera
   `promessi_sposi_chunks.json` sul desktop dell'utente.
2. Shell (PowerShell) copia il file in
   `middleware-qdrant-manager/data/corpus/promessi_sposi_chunks.json`.
3. `qdrant_ingest_file` con `file_path: "corpus/promessi_sposi_chunks.json"`,
   `source_id: "promessi-sposi"`, `batch_size: 8` → 39 chunk ingestati.

Nessun tool MCP per il passo 2, ed è corretto così.

---

## Perché NON aggiungiamo un CRUD file ai tool MCP

### 1. Il mount di `data/` è read-only dentro il container
```
"Source":"...\\middleware-qdrant-manager\\data","RW":false
```
Un tool di scrittura MCP richiede di flippare il mount a RW, indebolendo
un hardening che il progetto mantiene deliberatamente (container non-root
UID/GID 1000, root filesystem read-only). Non va reversato per
convenience.

### 2. Violazione di layer
Il MCP server è definito come "thin, stateless translation layer" che
espone la REST API del middleware. Il filesystem dell'host non è parte
della REST API del middleware — è operativa di deploy. Mescolando le due
cose il server MCP smette di essere un proxy puro e diventa anche file
manager: due responsabilità, due superfici di attacco.

### 3. Espansione della blast radius del token MCP
Oggi un `MCP_API_TOKENS` compromesso fa danno su Qdrant (grave, ma
recuperabile da backup). Con un CRUD file, lo stesso token diventa
scrittura/cancellazione filesystem sull'host. Il delete è irreversibile:
niente undo, niente cestino. È lo stesso ragionamento dietro al gate
`confirm` di `replace_source` e `recreate=true`: le operazioni
distruttive si rendono difficili di proposito.

### 4. Path traversal è il classico bug
`ingest_file` già valida path (`..` e assoluti rifiutati con 400), ma su
una superficie di sola lettura. Un CRUD generale moltiplica le occasioni
di sbagliare la validazione — e un path traversal in un tool di scrittura
è una vulnerabilità shaped-RCE, non un bug minore.

### Nota — `qdrant_export_collection` (2026-08-07)

I quattro argomenti sopra restano validi: sono scritti contro un tool CRUD
sui file guidato dal chiamante. L'export nativo è stato valutato contro
tutti e quattro e li supera:

1. **Mount RW** — non tocca `data/`, che resta `:ro`. Scrive su un mount
   dedicato `exports/`.
2. **Layer** — l'endpoint `POST /points/export` rende la scrittura parte
   della REST API del middleware, eseguita dal servizio che già possiede
   quel filesystem. Il server MCP resta un proxy puro.
3. **Blast radius** — nessun path, filename o contenuto dal chiamante,
   nessuna delete, nessuna sovrascrittura fuori da `exports/`. Un token
   compromesso può consumare disco con dump di collection che poteva già
   leggere via `qdrant_scroll_points`.
4. **Path traversal** — non applicabile: nessun input del chiamante
   raggiunge il path.

Dettagli in `docs/superpowers/specs/2026-08-07-collection-export-endpoint-design.md` §6.1.

### 5. `qdrant_ingest_chunks` copre già il caso MCP-native
Il gap di *capability* non esiste: esiste un gap di *convenience* sui
file grossi. Per contenuti inline c'è già `ingest_chunks`. Il caso
"file grosso già pronto" è un caso operativo dell'utente, non
dell'agente — ed è giusto che resti operativo (shell, scp, script di
deploy).

---

## Dove è documentato questo pattern

Il contenuto è stato integrato in cinque punti, in ordine di raggiungibilità
per un agente MCP-only:

1. **`docs://workflows/ingest`** (risorsa MCP live, sezione "Staging a file
   this server cannot write") — la superficie principale: scopribile via
   `mcp_read_resource` senza alcun accesso al repo.
2. **`instructions` del server** (`smart_qdrant_mcp/server.py`) — il tetto di
   4 MiB e il rimando alla risorsa, consegnati nell'handshake `initialize`,
   quindi visti prima di qualunque tool description.
3. **Prompt `qdrant_bootstrap_collection`** (`smart_qdrant_mcp/prompts.py`,
   passo 5) — il caso "file non ancora staged" e i suoi due fallback.
4. **`smart-qdrant-mcp-server/README.md`**, sezione "Ingest" — l'esempio
   completo copia + ingest, per chi legge il repo.
5. **`smart-qdrant-mcp-server/README.md`**, sezione "Operational limits
   worth knowing" — il tetto di 4 MiB già documentato lì prima di questo
   piano (era corretto anche prima: non è stato toccato, solo confermato).

`AGENTS.md` rimanda qui per il razionale esteso.

---

## Se in futuro la frizione si fa sentire molto

Valutare (in ordine di costo crescente):

1. **Niente** — tenere il pattern operativo sopra. Costo: zero.
   Beneficio: zero frizione per i casi già coperti.
2. **Tool write-only stretto** `stage_ingest_file(path, content)` con
   validazione paranoica (path relativo a `data/`, no `..`, no assoluti,
   estensione `.json`, size limit, no overwrite senza `confirm`). Copre
   il 90% del bisogno senza dare list/delete. **Richiede comunque il
   mount RW** — il costo di hardening c'è lo stesso.
3. **Endpoint nel middleware** (non nel MCP) `POST /ingest/stage` che
   scrive in `data/` con la sandbox che già esiste. Il MCP resta proxy
   puro. Anche qui serve RW mount.

**Non aggiungere mai** un delete o un list arbitrario su `data/` via MCP:
il rapporto beneficio/rischio è nettamente sfavorevole.
