# Conceptualizer

**directory structure:**  

 ├── 1_bird_eva  
 │ ├── bird-documents-for-annotations.pdf  
 │ ├── bird_evaluation.csv  
 │ ├── conceptualizer_bird_translations.csv  
 │ └── eflomal_bird_translations.csv  
 ├── 2_swadesh_eva  
 │ ├── conceptualizer_triples.csv  
 │ ├── eflomal_triples.csv  
 │ └── norare_triples.csv  
 ├── 3_stability_eva  
 │ └── stability_results.txt  
 ├── 4_similarity_eva  
 │ └── sim_matrix.csv  
 └── README.md  

## Single concept across all languages

***bird-documents-for-annotations.pdf***: the document provided for the linguist annotator

***bird_evaluation.csv***: the categories (linguistic + panlex) for all (1335-4)=1331 languages  

***conceptualizer_bird_translations.csv***: results of Conceptualizer: containing the (language, number of verses containing `bird', translations\&frequencies)   
  
***eflomal_bird_translations.csv***: results of Eflomal: containing the (language, number of verses containing `bird', translations\&frequencies)   
  

## Swadesh concepts

***norare_triples.csv***: csv table containing 582 available triples of (concept, language, translation(s)) from NoRaRe (as gold standard)  

***conceptualizer_triples.csv***: csv table containing 582 available triples of (concept, language, translation(s)) by Conceptualizer  

***eflomal_triples.csv***: csv table containing 582 available triples of (concept, language, translation(s)) by Eflomal  


## Concept stability

***stability_results.txt***: data of the concreteness and computed stability measure of 83 chosen focal concepts  
  

## Language Similarity

***sim_matrix.csv***: conceptual similarity (using Swadesh32) between any languages (1280 languages available) 