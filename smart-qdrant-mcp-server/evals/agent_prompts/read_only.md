Lavori con il server MCP "qdrant". Nella collection `mcp_eval` fai le verifiche qui sotto in sola lettura: non modificare nulla e non toccare altre collection.

1. Lo stack è pronto a servire richieste? Quale profilo di embedding è attivo?
2. Quali campi del payload di `mcp_eval` sono indicizzati, e con che tipo?
3. Quanti chunk ci sono per ciascun `doc_type`?
4. Quanti chunk hanno il campo `severity` vuoto o assente? E quanti design note sono stati pubblicati nel 2023?
5. Trova il chunk che parla della chiave di configurazione `query.max_candidates`: in che release è? Poi rileggilo direttamente tramite il suo id e dimmi la data di pubblicazione.
6. Cerca "how to recover from failed writes in the ingestion pipeline" due volte, una con il reranker e una senza, facendoti restituire solo il `chunk_id` di ogni risultato. L'ordine dei primi 5 cambia?
7. Elenca tutti i chunk del componente `query-engine` a pagine di 5 risultati, seguendo la paginazione fino alla fine. Quanti sono in totale?
8. Secondo la documentazione del server, che forme accetta un filtro per i campi vuoti? Rispondi in due righe citando la fonte.

Alla fine dai un riepilogo: per ogni punto la risposta e gli eventuali errori incontrati.
