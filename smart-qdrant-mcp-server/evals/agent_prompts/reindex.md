Lavori con il server MCP "qdrant". La collection `agenttest_reindex` contiene circa 2000 ticket operativi. Devi rigenerarne i vettori con un reindex, in modo sicuro. Non toccare nessun'altra collection.

1. Prima di partire, verifica che il reindex sia possibile e dimmi quanti punti verranno rigenerati.
2. Avvia il reindex.
3. Appena è partito, mettilo in pausa. Verifica che sia davvero in pausa e annota quanti punti erano già stati elaborati.
4. Mentre è in pausa, prova ad aggiungere alla collection un documento nuovo: "Pause test document." con team=platform. Riporta cosa succede e perché. Non annullare il job per aggirare un eventuale rifiuto.
5. Riprendi il reindex e verifica che riparta dal punto in cui si era fermato, non da zero.
6. Aspetta che sia davvero finito.
7. Alla fine verifica che il numero di punti sia invariato, che la collection sia valida per il profilo di embedding attivo, e che una ricerca per "gateway throttle after a version upgrade" restituisca risultati.
8. Non cancellare la collection: alla pulizia penso io.

Alla fine dai un riepilogo passo per passo: cosa hai fatto, l'esito, e gli eventuali errori.
