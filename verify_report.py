"""Verify that the submitted WPS DOCX contains the exact measured headline metrics."""
import json
from pathlib import Path
from docx import Document

root = Path(__file__).parent
report = root / 'CNN结构修改与场景适配实验报告_WPS.docx'
doc = Document(report)
text = '\n'.join(p.text for p in doc.paragraphs)
text += '\n' + '\n'.join(c.text for t in doc.tables for row in t.rows for c in row.cells)
for name in ('baseline','depthwise','se','depthwise_se'):
    result = json.loads((root / 'results' / f'{name}.json').read_text(encoding='utf-8'))
    accuracy = f'{result["test"]["accuracy"]*100:.2f}%'
    parameters = f'{result["parameters"]:,}'
    assert accuracy in text, (name, accuracy)
    assert parameters in text, (name, parameters)
    assert result['test']['correct'] == round(result['test']['accuracy'] * 10000)
url_file = root / 'repository_url.txt'
if url_file.exists():
    assert url_file.read_text(encoding='utf-8').strip() in text
print('PASS: four test accuracies, parameter counts, and correct counts match output JSON')
