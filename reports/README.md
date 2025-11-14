Report generation
=================

This folder contains helpers and guidance to produce shareable reports from the cleaned purchase data.

Files
- generate_report.py — simple script that creates an HTML report (charts + key metrics) and writes to `reports/report.html`.

How to generate a report (recommended)
1. Install dependencies (see project `requirements.txt`):

   pip install -r requirements.txt

2. Run the script:

   python reports/generate_report.py

3. Open `reports/report.html` in your browser.

Notes and next steps
- For production reports consider using `nbconvert` (from a Jupyter notebook) or `WeasyPrint` / `wkhtmltopdf` to produce PDFs.
- You can integrate report generation into a CI job to produce periodic reports.
