# Bird Classifier

## Setup
pip install -r requirements.txt

## Run
python src/data/ingestion.py --input data.zip

### Revisar images de review
```
python scripts/review_tool.py --source review
```
### Solo las de baja confianza
```
python scripts/review_tool.py --source review --subdir low_confidence
```
### Darle segunda oportunidad a rechazadas
```
python scripts/review_tool.py --source rejected --subdir no_detection
```
### Ver cuántas hay pendientes sin abrir la ventana
```
python scripts/review_tool.py --source review --list
```
```
python review_tool.py --source processed
```
```
python review_tool.py --source processed --subdir Turdus_fuscater
```