# La memoria n-gram di Qwen3.8-Flash-Next, e come la replichiamo in piccolo

Studio del 4 novembre 2026 per la linea A del Fast Lane
(`docs/cineca/fastlane-plan.md`). Fonte: il codice del modello in
transformers 5.22.1 (`models/qwen4_exp/modeling_qwen4_exp.py`, classi
`Qwen4ExpTextNGramEmbedding` e `Qwen4ExpTextPLELayer`) e il `config.json` di
`Qwen/Qwen3.8-Flash-Next`. La nostra replica è `forge/eullm_forge/memory/ngram.py`;
l'addestramento è `forge/scripts/ngram_train.py`.

## Cosa fa Flash-Next

**Dove.** Un solo strato di memoria (`ple_layer_ids: [2]`): il decoder layer
di indice 1 la somma al residuo **prima** della propria attenzione, cioè
subito dopo il primo strato. È una lettura lessicale molto bassa, prima che
il modello abbia fatto quasi niente.

**Quali n-gram.** 2-gram e 3-gram (`ngram_size: 3`), 8 teste per ordine
(`heads_per_ngram: 8`): 16 teste in tutto. Ogni testa ha una tabella sua di
dimensione prima, i primi consecutivi dopo 20 milioni
(`ngram_vocab_size_base`), così due teste non condividono mai il modulo e
una collisione in una testa quasi mai si ripete nelle altre.

**L'hash.** Per ogni posizione, i token di 0, 1 e 2 posti indietro,
ciascuno moltiplicato per un moltiplicatore dispari fissato dal seme
(splitmix64, limitato perché il prodotto stia in un int64), combinati con
XOR, poi modulo il primo della testa più l'offset della testa nella tabella
unica. I moltiplicatori dipendono solo dalla posizione (0, 1 o 2 posti
indietro), non dalla testa: le 8 teste di un ordine differiscono solo per il
primo.

**I confini dei documenti.** Lo spostamento all'indietro si ferma all'EOS:
le posizioni prima dell'inizio del documento valgono EOS, quindi un n-gram
non attraversa mai due documenti impacchettati nella stessa riga. Le
posizioni di padding sono messe a EOS prima dell'hash e azzerate dopo il
gate (`apply_mask_to_padding_states`).

**La tabella.** 16 × ~20 M = 320 milioni di righe, ognuna larga
`ple_embed_dim / 16` = 2560 / 16 = 160. **Circa 51 miliardi di parametri**,
più di un quarto dei 180 del modello. Il peso è escluso dal `device_map`
(`_no_placement_params`): può stare in RAM della CPU, perché ogni token ne
legge 16 righe e basta.

**La lettura.** Le 16 righe concatenate (2560) vanno in:
* una *value* condivisa, proiettata a `hidden_size` (2560);
* una *key* per ognuno dei 4 flussi del residuo (`hc_count: 4`, le
  hyper-connection), con RMSNorm per flusso;
* il gate di ogni flusso è `sigmoid(sign · sqrt|k·q / √H|)`, con `q` il
  flusso normalizzato: la memoria entra solo dove il contesto la "riconosce";
* sopra il valore filtrato c'è una convoluzione depthwise **dilatata**
  (kernel 4, dilatazione 3 = `ngram_size`, SiLU) su una versione
  normalizzata: guarda gli n-gram di 3, 6 e 9 posizioni prima;
* uscita = valore filtrato + convoluzione, sommata ai flussi.

**In generazione.** Gli ultimi due token stanno nella cache come uno stato
di convoluzione (`state_idx=2`), la storia della convoluzione dilatata in
un altro (`state_idx=1`): decodificare un token alla volta dà gli stessi
hash dell'intera sequenza.

Non c'è un fattore di scala: la tabella è stata addestrata da zero insieme
al resto del modello.

## La nostra replica in piccolo

| | Flash-Next | A2-mini (`sbatch_ngram_train.slurm`) |
|---|---|---|
| modello | addestrato con la memoria | **congelato**, memoria innestata dopo |
| strato | prima del layer 1 (dopo il layer 0) | dopo il layer 1 (`--layer 1`) |
| ordini, teste | 2+3, 8 per ordine | uguale |
| righe per testa | ~20 M (primi) | 400.000 (primi) |
| larghezza per testa | 160 | 64 (`--embed-dim 1024`) |
| parametri della tabella | ~51 miliardi | **0,41 miliardi** (1/125) |
| hash | prodotti in XOR, mod primo | prodotti sommati, mod primo |
| flussi del residuo | 4 (una key per flusso) | 1 (r2l non ha hyper-connection) |
| gate | `sigmoid(sign·sqrt|k·q/√H|)` | uguale |
| convoluzione | depthwise, kernel 4, dilatazione 3, SiLU | uguale |
| uscita | somma diretta | × `scale`, **inizializzato a 0** |
| padding | EOS, azzerato dopo il gate | token fuori vocabolario, azzerato uguale |
| confine dei documenti | l'n-gram si ferma all'EOS | **attraversa l'EOS** (da sistemare, sotto) |
| ottimizzatore | — | SparseAdam sulla tabella, AdamW sul resto |

Il fattore di scala a zero è la differenza che conta: innestata su un
modello già addestrato, la memoria all'inizio non cambia nulla
(`test_at_scale_zero_the_model_is_unchanged`). Il modello guadagna solo se
l'addestramento trova utile aprirla, e il controllo LoRA con gli stessi
parametri dice se la stessa capacità spesa altrove avrebbe reso di più.

## Da cambiare prima di A3

1. **I confini dei documenti.** Nelle righe impacchettate i primi due token
   di ogni articolo si hashano con la coda dell'articolo precedente. Sono
   due posizioni per documento, quindi rumore piccolo in A2-mini, ma non è
   quello che fa Flash-Next. Si sistema in `row_ids` con lo stesso
   spostamento che si ferma all'EOS, con un test che due documenti nella
   stessa riga diano gli hash che darebbero separati.
2. **Lo strato.** `--layer 0` è la posizione di Flash-Next: in A3 vanno
   provati 0 e 1, è una delle domande della curva taglia/beneficio.
3. **XOR invece della somma**, solo se si vuole poter caricare le tabelle
   nel formato di Flash-Next; per le collisioni le due combinazioni si
   equivalgono.
4. **La stessa tabella letta a due strati**, per esempio dopo il layer 1 e
   più in alto. Le righe lette sono le stesse (l'hash dipende solo dai
   token), ma key, gate e convoluzione di ogni lettura sono propri e il
   gate confronta la memoria con uno stato del modello più elaborato: la
   seconda lettura può prendere ciò che la prima ha scartato. Flash-Next
   legge una volta sola. L'idea viene da LoOM (Yang et al., *Loop-Invariant
   Memory for Causal Encoder-Decoder Language Models*, arXiv 2611.06835,
   4 novembre 2026, sezione 5.4): lì ripetere uno strato che legge la
   memoria globale con lo stato già raffinato le fa scegliere voci nuove (il
   32% dell'attenzione della quarta visita cade su voci che la prima non
   aveva scelto), e congelare le scelte della prima visita costa 0,011 nat.
   Costo per noi: una seconda proiezione e un secondo gate, nessuna riga di
   tabella in più. Da provare in A3 solo se A2 passa il cancello.

Il resto di LoOM non riguarda la linea A: la sua memoria è la cache del
contesto, non una tabella di parametri, e non aggiunge capacità di sapere;
si addestra da zero con la sua architettura (204 milioni di parametri, 5
miliardi di token, un seme), e il guadagno (0,020 nat, +0,79 punti) costa
il doppio della latenza a 21 passaggi. L'idea di un solo modello con più
budget di calcolo scelti all'uso interesserebbe l'engine, ma richiede
un'architettura che llama.cpp oggi non ha.

## Cosa misurare su Flash-Next stesso (due nodi, bf16)

1. **Quanto conta la sua memoria sul diritto italiano.** Lo stesso esame a
   libro chiuso con l'uscita del PLE azzerata e no: se il punteggio crolla
   senza, è la prova che una tabella n-gram porta conoscenza di dominio, e
   non solo forma della lingua.
2. **Riaddestrare solo la tabella** sul corpus memoria degli articoli, con
   tutto il resto congelato: è E3 (la riforma simulata) sul modello che ha
   la memoria per costruzione. Sono 51 miliardi di parametri con gradienti
   sparsi: l'ottimizzatore va tenuto in CPU, ed è da provare prima su un
   nodo con un sottoinsieme di teste.
3. **Le collisioni.** Su 100 milioni di token del corpus legale, quante
   righe distinte legge ogni testa e con quanti n-gram diversi le divide: è
   il numero che dice quante righe servono davvero alla nostra versione
   piccola.

## Misurato (12 novembre 2026)

Il primo esperimento qui sopra, fatto su Leonardo con la GGUF Q4 (la
proiezione dei valori dello strato PLE azzerata, `gguf_zero_tensors.py`):
sul libro chiuso Flash-Next scende da 247 a 227 risposte giuste su 716, e
ogni colonna di conoscenza cala di 3-8 punti (pene dal 29% al 21%, articoli
veri riconosciuti dal 24% al 16%). La tabella porta una parte della
conoscenza, non la maggior parte: senza, il modello resta molto sopra tutti
gli altri misurati. Dettagli in `docs/legal-it-experiments.md`, sezione 17.
