# Conceptualizer

**IMPORTANT:** Please refer to [ColexificationNet](https://github.com/yihongL1U/ColexificationNet) for a more effcient implemetation of Conceptualizer.

**directory structure:**  

```
.
├── 1_bird_eva
│   ├── bird-documents-for-annotations.pdf
│   ├── bird_evaluation.csv
│   ├── conceptualizer_bird_translations.csv
│   └── eflomal_bird_translations.csv
├── 2_swadesh_eva
│   ├── conceptualizer_triples.csv
│   ├── eflomal_triples.csv
│   └── norare_triples.csv
├── 3_stability_eva
│   └── stability_results.txt
├── 4_similarity_eva
│   └── sim_matrix.csv
├── README.md
├── concept_script.py
├── concepts
│   ├── all_concepts.txt
│   ├── bible_concepts.txt
│   └── swadesh_concept.txt
├── conceptualizer.py
├── parse_pbc.py
├── pbc_table.csv
├── uroman.py
└── utils.py
```

## Run Conceptualizer

The conceptualizer is implemented in `./conceptualizer.py`.  

Run conceptualizer on selected 32 Swadesh concepts using the following command:  

```
export PYTHONIOENCODING=utf8; nohup python -u ./concept_script.py --src_lang eng --concept_path "./concepts/swadesh_concepts.txt" --tgt_langs "all" > ./temp/log_run_all_swadesh.txt 2>&1 &
```

Run conceptualizer on selected 51 Bible concepts using the following command:

```
export PYTHONIOENCODING=utf8; nohup python -u ./concept_script.py --src_lang eng --concept_path "./concepts/bible_concepts.txt" --tgt_langs "all" > ./temp/log_run_all_bible.txt 2>&1 &
```

Run conceptualizer on all 83 concepts (it can take **very long** time) using the following command:

```
export PYTHONIOENCODING=utf8; nohup python -u ./concept_script.py --src_lang eng --concept_path "./concepts/all_concepts.txt" --tgt_langs "all" > ./temp/log_run_all_bible.txt 2>&1 &
```

## Evaluations

### Single concept across all languages

***bird-documents-for-annotations.pdf***: the document provided for the linguist annotator

***bird_evaluation.csv***: the categories (linguistic + panlex) for all (1335-4)=1331 languages  

***conceptualizer_bird_translations.csv***: results of Conceptualizer: containing the (language, number of verses containing `bird', translations & frequencies)   
  
***eflomal_bird_translations.csv***: results of Eflomal: containing the (language, number of verses containing `bird', translations & frequencies)   
  

### Swadesh concepts

***norare_triples.csv***: csv table containing 582 available triples of (concept, language, translation(s)) from NoRaRe (as gold standard)  

***conceptualizer_triples.csv***: csv table containing 582 available triples of (concept, language, translation(s)) by Conceptualizer  

***eflomal_triples.csv***: csv table containing 582 available triples of (concept, language, translation(s)) by Eflomal  


### Concept stability

***stability_results.txt***: data of the concreteness and computed stability measure of 83 chosen focal concepts  
  

### Language Similarity

***sim_matrix.csv***: conceptual similarity (using Swadesh32) between any languages (1280 languages available) 

## References

Please cite if you found the resources in this repository useful.