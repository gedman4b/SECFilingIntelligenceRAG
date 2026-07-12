**Backend implementation (FastAPI)**

The FastAPI backend implements the six agent stages from Part 4. This section shows the core code with commentary. A working version of the code is provided alongside this document.
Project structure
backend/
  app/
    main.py                 # FastAPI app, /query and /ingest endpoints
    schemas.py              # Pydantic models (Plan, Fact, Response)
    agents/
      planner.py            # Query Planner Agent (LLM)
      fact_retriever.py     # Deterministic SQL against fact store
      numerical_reasoner.py # Deterministic arithmetic
      prose_retriever.py    # Vector search over prose only
      verifier.py           # Consistency checks + warnings
      answer_composer.py    # Narrative composition (LLM)
    ingest/
      pdfs/                 # repository of SEC filing pdf documents
      pdf_parser.py         # pdfplumber-based layout-aware parsing
      table_classifier.py   # LLM classification of parsed tables
      fact_extractor.py     # Extract typed facts from tables
      canonicalizer.py      # Map raw labels to canonical metric IDs
    store/
      db.py                 # SQLite (Postgres in prod)
      vector_store.py       # Chroma for prose embeddings
    eval/
      benchmark.py          # Hand-verified Q/A pairs
      runner.py             # Evaluation harness
  requirements.txt

**Frontend implementation (Node.js)**
The Node.js frontend is a lightweight Express server that serves a single-page UI. The UI's primary job is to make traceability visible. Every answer is displayed alongside its raw values, calculation steps, and clickable citations. A skeptical finance user can verify the answer without leaving the page.
Project structure
frontend/
  server.js               # Express server serving static + proxy to FastAPI
  public/
    index.html            # Single-page UI
    styles.css            # Minimal styling
    app.js                # Query submission + response rendering
  package.json
