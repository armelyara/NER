# Analyse du Biais de Nom

## Statistiques Générales

- Entités uniques: 1888
- Entités ambiguës: 146
- Taux d'ambiguïté: 7.7%

## Top 10 Entités Ambiguës

| Entité | Types observés |
|--------|----------------|
| Bristol | LOCATION, ORGANIZATION, PERSON |
| Rollins | LOCATION, ORGANIZATION, PERSON |
| Malone | LOCATION, ORGANIZATION, PERSON |
| Mesta | LOCATION, ORGANIZATION |
| Sunderland | LOCATION, PERSON |
| Ivanhoe | LOCATION, PERSON |
| Stroh | LOCATION, ORGANIZATION |
| Kohler | LOCATION, ORGANIZATION |
| Casablanca | ORGANIZATION, PERSON |
| Atlas | LOCATION, PERSON |

## Top 5 Erreurs Fréquentes (BERT)

| Gold | Prédiction | Fréquence |
|------|------------|----------|
| PERSON | ORGANIZATION | 427 |
| ORGANIZATION | PERSON | 248 |
| LOCATION | PERSON | 234 |
| LOCATION | ORGANIZATION | 224 |
| PERSON | LOCATION | 223 |
