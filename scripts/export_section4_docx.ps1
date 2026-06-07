param(
    [string]$SourceMarkdown = "docs\paper1\final_package\section4_experimental_evaluation_elsevier.md",
    [string]$OutputDocx = "docs\paper1\final_package\section4_experimental_evaluation_elsevier.docx"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Utf8File {
    param(
        [string]$Path,
        [string]$Value
    )
    [System.IO.File]::WriteAllText($Path, $Value, (New-Object System.Text.UTF8Encoding($false)))
}

function Escape-XmlText {
    param([string]$Text)
    if ($null -eq $Text) { return "" }
    return [System.Security.SecurityElement]::Escape($Text)
}

function New-ParagraphXml {
    param(
        [string]$Text,
        [string]$Style = "Normal"
    )
    $safe = Escape-XmlText $Text
    return "<w:p><w:pPr><w:pStyle w:val=`"$Style`"/></w:pPr><w:r><w:t xml:space=`"preserve`">$safe</w:t></w:r></w:p>"
}

function Convert-MarkdownToWordXml {
    param([string[]]$Lines)

    $paragraphs = New-Object System.Collections.Generic.List[string]
    $buffer = New-Object System.Collections.Generic.List[string]

    function Flush-Buffer {
        if ($buffer.Count -gt 0) {
            $text = ($buffer -join " ").Trim()
            if ($text.Length -gt 0) {
                $paragraphs.Add((New-ParagraphXml -Text $text -Style "Normal"))
            }
            $buffer.Clear()
        }
    }

    foreach ($rawLine in $Lines) {
        $line = $rawLine.TrimEnd()

        if ([string]::IsNullOrWhiteSpace($line)) {
            Flush-Buffer
            continue
        }

        if ($line.StartsWith("# ")) {
            Flush-Buffer
            $paragraphs.Add((New-ParagraphXml -Text $line.Substring(2).Trim() -Style "Heading1"))
            continue
        }
        if ($line.StartsWith("## ")) {
            Flush-Buffer
            $paragraphs.Add((New-ParagraphXml -Text $line.Substring(3).Trim() -Style "Heading2"))
            continue
        }
        if ($line.StartsWith("### ")) {
            Flush-Buffer
            $paragraphs.Add((New-ParagraphXml -Text $line.Substring(4).Trim() -Style "Heading3"))
            continue
        }
        if ($line.StartsWith("- ")) {
            Flush-Buffer
            $paragraphs.Add((New-ParagraphXml -Text ("• " + $line.Substring(2).Trim()) -Style "Normal"))
            continue
        }

        $buffer.Add($line.Trim())
    }

    Flush-Buffer
    return ($paragraphs -join "")
}

$sourcePath = Join-Path (Get-Location) $SourceMarkdown
$outputPath = Join-Path (Get-Location) $OutputDocx
$buildRoot = Join-Path (Get-Location) "temp_section4_docx_build"

if (-not (Test-Path $sourcePath)) {
    throw "Source markdown not found: $sourcePath"
}

if (Test-Path $buildRoot) {
    Remove-Item $buildRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $buildRoot "_rels") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $buildRoot "docProps") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $buildRoot "word") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $buildRoot "word\_rels") | Out-Null

$lines = Get-Content $sourcePath
$bodyXml = Convert-MarkdownToWordXml -Lines $lines

$contentTypes = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
'@

$rels = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
'@

$docRels = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships" />
'@

$coreXml = @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Section 4 Experimental Evaluation - Elsevier Revision</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">$(Get-Date -Format s)Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">$(Get-Date -Format s)Z</dcterms:modified>
</cp:coreProperties>
"@

$appXml = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
</Properties>
'@

$stylesXml = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="24"/></w:rPr>
  </w:style>
</w:styles>
'@

$documentXml = @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    $bodyXml
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"@

Write-Utf8File -Path (Join-Path $buildRoot "[Content_Types].xml") -Value $contentTypes
Write-Utf8File -Path (Join-Path $buildRoot "_rels\.rels") -Value $rels
Write-Utf8File -Path (Join-Path $buildRoot "docProps\core.xml") -Value $coreXml
Write-Utf8File -Path (Join-Path $buildRoot "docProps\app.xml") -Value $appXml
Write-Utf8File -Path (Join-Path $buildRoot "word\document.xml") -Value $documentXml
Write-Utf8File -Path (Join-Path $buildRoot "word\styles.xml") -Value $stylesXml
Write-Utf8File -Path (Join-Path $buildRoot "word\_rels\document.xml.rels") -Value $docRels

$zipPath = [System.IO.Path]::ChangeExtension($outputPath, ".zip")
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}
if (Test-Path $outputPath) {
    Remove-Item $outputPath -Force
}

Compress-Archive -Path (Join-Path $buildRoot "*") -DestinationPath $zipPath -Force
Move-Item $zipPath $outputPath

Remove-Item $buildRoot -Recurse -Force
Write-Output $outputPath
