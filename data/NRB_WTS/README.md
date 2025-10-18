Overview
============

This repository contains the data and source code used in the following research publication

    Abbas Ghaddar, Philippe Langlais, Ahmad Rashid, Mehdi Rezagholizadeh
    Context-aware Adversarial Training for Name Regularity Bias in Named Entity Recognition
    TACL approved
    
Data Format
============
the directory contains NRB and Witness (WTS) sets for 7 languages, original is English while other languages are produced via MT.  
The files are named as follow: `{lang}.{portion}.{format}` where:

```
lang = en|de|es|nl|af|hr|fi|da
portion = nrb|wts
format = conll|txt
```

We provide the same `{lang}.{portion}` in two formats: `.conll` (tokenized sentence) and `.txt` (untokenized sentence).
In the latter, entities are marked within the sentence as follow: 

```
[Bali]_{ORGANIZATION} merged with Hanes Corporation in 1969.     
```

Evaluation
============

Each file contains a set of sentences where where each sentence contains **one named entity annotation**.
The list of tags includes: `PERSON, LOCATION, ORGANIZATION`.

Since  in NRB and WTS, there is only one entity per sentence to annotate, 
a system is evaluated on its ability to correctly identify the boundaries 
of this entity and its type. That is, a system receives `1` if the entity of concern is correctly predicted 
regardless of other entities that maybe identified by the system.
 