param(
  [string]$Source = (Join-Path $PSScriptRoot 'CNN结构修改与场景适配实验报告.docx'),
  [string]$DocxOut = (Join-Path $PSScriptRoot 'CNN结构修改与场景适配实验报告_WPS.docx'),
  [string]$PdfOut = (Join-Path $PSScriptRoot 'report_qa.pdf')
)
$app = New-Object -ComObject KWPS.Application
$app.Visible = $false
try {
  $document = $app.Documents.Open($Source)
  try {
    $document.SaveAs2($DocxOut, 12)
    $document.ExportAsFixedFormat($PdfOut, 17)
  } finally {
    $document.Close()
  }
} finally {
  $app.Quit()
}
Get-Item -LiteralPath $DocxOut, $PdfOut | Select-Object FullName, Length
