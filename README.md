# ⛏️ AI-Analysetool voor Technische Mijnbouwrapporten

Een Python-tool die ongestructureerde technische rapporten uit de mijnbouwsector automatisch analyseert en omzet in gestructureerde data met behulp van Large Language Models (LLM's) en NLP.

De app kan online worden geraadpleegd via volgend link: https://mining-analyst-ai-aefzjidjgyd6q3efnciut9.streamlit.app/

## 🎯 Doel van het project
Technische rapporten in de mijnbouw (bv. `NI 43-101` rapporten) bevatten cruciale geologische en metallurgische data, maar zijn vaak honderden pagina's lang en ongestructureerd (PDF). Handmatige analyse is tijdrovend en foutgevoelig.

Deze tool automatiseert dit proces om snel inzichten te genereren in:
- Ertsgraden en samenstellingen.
- Locaties en afzettingen.
- Operationele parameters.

## 🚀 Belangrijkste Functionaliteiten
- **PDF Data Extractie:** Geautomatiseerd inlezen van tekst en tabellen uit ruwe PDF-bestanden.
- **AI-Interpretatie:** Gebruik van LLM's om context-afhankelijke informatie te begrijpen (bijv. onderscheid maken tussen 'verwachte' en 'bewezen' reserves).
- **Gestructureerde Output:** Export van resultaten naar Excel/CSV/JSON voor verdere analyse.

## 🛠️ Tech Stack
* **Taal:** Python 3.9+
* **AI & NLP:** [Bijv. OpenAI API / LangChain / HuggingFace transformers]
* **Data Processing:** Pandas, NumPy, [PyPDF2 / PDFPlumber]
* **Visualisatie:** Matplotlib / Seaborn
