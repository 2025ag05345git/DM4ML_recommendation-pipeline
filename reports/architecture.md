# Pipeline architecture

```mermaid
flowchart LR
    A[Seeded synthetic generator] --> B[Raw CSV users, 5,000 movies, ratings]
    B --> C[Schema, language, quality and FK validation]
    C -->|valid| D[Preparation and chronological split]
    C -->|invalid| E[Quarantine CSV and validation report]
    D --> F[Matrix, activity, popularity]
    D --> G[Reusable user, movie, event features]
    F --> H[Popularity baseline]
    G --> I[Personalized genre content model]
    H --> J[Precision@5, Recall@5, NDCG@5, coverage, diversity]
    I --> J
    J --> K[Comparison, recommendations and run tracking]
    K --> L[Markdown/PDF report]
    M[Dependency-checked scheduled Python workflow] --> A
    M --> C
    M --> D
    M --> F
    M --> G
    M --> H
    M --> I
    M --> J
    N[Future PostgreSQL adapter] -. disabled placeholder .-> B
```

The workflow communicates through CSV and JSON artifacts. PostgreSQL is
documented as a future storage option but is not used in the active run.
