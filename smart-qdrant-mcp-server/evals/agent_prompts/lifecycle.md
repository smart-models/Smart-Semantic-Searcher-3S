Lavori con il server MCP "qdrant". Esegui un ciclo di vita completo su una collection usa e getta chiamata `agenttest_lifecycle`. Non toccare nessun'altra collection.

1. Crea la collection. Se esiste già, fermati e segnalalo.
2. Rendi filtrabili i campi `lang` (stringa) e `year` (intero).
3. Carica questi 4 documenti come sorgente `src-a` (l'identificativo di ciascuno è il numero):
   1: "Il reranker riordina i risultati della ricerca ibrida." lang=it, year=2024
   2: "Sparse vectors capture exact identifiers like ERR-4021." lang=en, year=2019
   3: "La quantizzazione riduce la memoria dei vettori densi." lang=it, year=2018
   4: "Payload indexes make filters fast in strict mode." lang=en, year=2023
4. Aggiungi un quinto documento singolo senza ricaricare gli altri: "Hybrid search fuses dense and sparse rankings." lang=en, year=2025.
5. Imposta `reviewed=true` su tutti i documenti con lang=it.
6. Togli la chiave `reviewed` da uno solo di quei documenti.
7. Prova a cancellare la collection passando come conferma il suo nome scritto TUTTO MAIUSCOLO: deve essere rifiutato. Verifica che la collection esista ancora con lo stesso numero di punti.
8. Cancella i documenti con year < 2020, dicendomi prima quanti sono.
9. Verifica lo stato della collection, poi attiva la quantizzazione senza ricrearla.
10. Esporta la collection e dimmi dove è stato scritto il file.
11. Cancella la collection e verifica che non esista più.

Alla fine dai un riepilogo passo per passo: cosa hai fatto, l'esito, e gli eventuali errori.
