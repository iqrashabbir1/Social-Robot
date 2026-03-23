Add-Type -AssemblyName System.Drawing

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FigureDir = Join-Path $ProjectRoot "outputs\figures"
$TableDir = Join-Path $ProjectRoot "outputs\tables"
$CsvDir = Join-Path $ProjectRoot "outputs\csv"

function New-Canvas {
    param(
        [int]$Width,
        [int]$Height
    )

    $bitmap = New-Object System.Drawing.Bitmap $Width, $Height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $graphics.Clear([System.Drawing.Color]::White)
    return @{ Bitmap = $bitmap; Graphics = $graphics }
}

function Save-Canvas {
    param(
        $Canvas,
        [string]$Path
    )

    $Canvas.Bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $Canvas.Graphics.Dispose()
    $Canvas.Bitmap.Dispose()
}

function New-Brush {
    param([string]$Hex)
    return New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml($Hex))
}

function Draw-Label {
    param(
        $Graphics,
        [string]$Text,
        [float]$X,
        [float]$Y,
        [float]$Size = 14,
        [string]$Hex = "#1f2933",
        [bool]$Bold = $false
    )

    $style = if ($Bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    $font = New-Object System.Drawing.Font("Segoe UI", $Size, $style)
    $brush = New-Brush $Hex
    $Graphics.DrawString($Text, $font, $brush, $X, $Y)
    $font.Dispose()
    $brush.Dispose()
}

function Draw-RoundedBox {
    param(
        $Graphics,
        [float]$X,
        [float]$Y,
        [float]$Width,
        [float]$Height,
        [string]$FillHex,
        [string]$BorderHex = "#ffffff"
    )

    $fill = New-Brush $FillHex
    $border = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml($BorderHex), 2)
    $Graphics.FillRectangle($fill, $X, $Y, $Width, $Height)
    $Graphics.DrawRectangle($border, $X, $Y, $Width, $Height)
    $fill.Dispose()
    $border.Dispose()
}

function Score-Color {
    param([int]$Score)
    switch ($Score) {
        0 { return "#f7fbff" }
        1 { return "#d0e1f2" }
        2 { return "#abd0e6" }
        default { return "#2166ac" }
    }
}

function Generate-LiteratureHeatmap {
    $rows = Import-Csv (Join-Path $TableDir "literature_comparison_matrix.csv")
    $columns = @(
        "predictive_health_risk_capability",
        "emotion_awareness",
        "medication_adherence_support",
        "ros2_digital_twin_support",
        "iot_edge_deployment",
        "explainability",
        "hitl_safety_governance",
        "privacy_awareness",
        "telepresence_cultural_adaptation",
        "real_world_readiness"
    )
    $labels = @("Risk","Emotion","Adherence","ROS2/Twin","IoT/Edge","Explain","HITL","Privacy","Telepresence","Readiness")
    $canvas = New-Canvas -Width 1400 -Height 650
    $g = $canvas.Graphics

    Draw-Label $g "Literature-Aligned Capability Coverage" 55 35 24 "#133c55" $true
    Draw-Label $g "Qualitative rubric, 0 to 3. Source: outputs/tables/literature_comparison_matrix.csv" 55 70 12 "#52606d"

    for ($i = 0; $i -lt $labels.Count; $i++) {
        Draw-Label $g $labels[$i] (170 + 100 * $i) 120 11 "#334e68" $true
    }

    for ($row = 0; $row -lt $rows.Count; $row++) {
        Draw-Label $g $rows[$row].approach_id 70 (170 + 50 * $row) 14 "#133c55" $true
        for ($col = 0; $col -lt $columns.Count; $col++) {
            $score = [int]$rows[$row].($columns[$col])
            Draw-RoundedBox $g (155 + 100 * $col) (155 + 50 * $row) 82 32 (Score-Color $score) "#d9e2ec"
            Draw-Label $g "$score" (190 + 100 * $col) (161 + 50 * $row) 12 ($(if ($score -eq 3) { "#ffffff" } else { "#1f2933" })) $true
        }
    }

    Save-Canvas $canvas (Join-Path $FigureDir "literature_gap_heatmap.png")
}

function Generate-LiteratureBar {
    $rows = Import-Csv (Join-Path $TableDir "literature_comparison_matrix.csv")
    $columns = @(
        "predictive_health_risk_capability",
        "emotion_awareness",
        "medication_adherence_support",
        "ros2_digital_twin_support",
        "iot_edge_deployment",
        "explainability",
        "hitl_safety_governance",
        "privacy_awareness",
        "telepresence_cultural_adaptation",
        "real_world_readiness"
    )
    $canvas = New-Canvas -Width 1200 -Height 650
    $g = $canvas.Graphics
    $axisPen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml("#334e68"), 2)

    Draw-Label $g "Integrated Capability Comparison Across Approaches" 55 35 24 "#133c55" $true
    Draw-Label $g "Higher bars indicate broader stack coverage, not measured benchmark performance." 55 70 12 "#52606d"
    $g.DrawLine($axisPen, 90, 560, 1110, 560)
    $g.DrawLine($axisPen, 90, 120, 90, 560)

    for ($i = 0; $i -le 30; $i += 5) {
        $y = 560 - ($i * 13)
        $gridPen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml("#d9e2ec"), 1)
        $g.DrawLine($gridPen, 90, $y, 1110, $y)
        Draw-Label $g "$i" 45 ($y - 8) 11 "#486581"
        $gridPen.Dispose()
    }

    for ($i = 0; $i -lt $rows.Count; $i++) {
        $total = 0
        foreach ($column in $columns) { $total += [int]$rows[$i].$column }
        $height = $total * 13
        $x = 130 + ($i * 110)
        $y = 560 - $height
        $fill = if ($rows[$i].approach_id -eq "A8") { "#2a9d8f" } else { "#5c677d" }
        Draw-RoundedBox $g $x $y 65 $height $fill $fill
        Draw-Label $g $rows[$i].approach_id ($x + 18) 575 12 "#1f2933" $true
        Draw-Label $g "$total" ($x + 18) ($y - 22) 12 "#1f2933" $true
    }

    $axisPen.Dispose()
    Save-Canvas $canvas (Join-Path $FigureDir "literature_radar_or_bar_comparison.png")
}

function Generate-ArchitectureFigure {
    $rows = Import-Csv (Join-Path $CsvDir "system_module_map.csv")
    $canvas = New-Canvas -Width 1400 -Height 780
    $g = $canvas.Graphics

    Draw-Label $g "System Architecture Overview" 55 35 24 "#133c55" $true
    Draw-Label $g "Source: outputs/csv/system_module_map.csv" 55 70 12 "#52606d"

    $layerColors = @{
        "Layer 1" = "#133c55"
        "Layer 2" = "#2a9d8f"
        "Layer 3" = "#e9c46a"
        "Layer 4" = "#f4a261"
        "Layer 5" = "#e76f51"
    }
    $xMap = @{
        "Layer 1" = 70
        "Layer 2" = 340
        "Layer 3" = 610
        "Layer 4" = 880
        "Layer 5" = 1150
    }
    $offsets = @{}

    foreach ($row in $rows) {
        if (-not $offsets.ContainsKey($row.layer)) { $offsets[$row.layer] = 0 }
        $x = $xMap[$row.layer]
        $y = 145 + ($offsets[$row.layer] * 95)
        $offsets[$row.layer] += 1
        Draw-RoundedBox $g $x $y 210 70 $layerColors[$row.layer] "#ffffff"
        Draw-Label $g $row.module_id ($x + 15) ($y + 10) 15 "#ffffff" $true
        Draw-Label $g $row.module_name ($x + 15) ($y + 35) 12 "#ffffff"
    }

    Draw-RoundedBox $g 100 520 400 120 "#f8fafc" "#cbd2d9"
    Draw-Label $g "Preserved Baseline Path" 125 545 20 "#133c55" $true
    Draw-Label $g "camera -> DeepFace -> speech SVM -> rule fusion -> response -> TTS" 125 580 13 "#334e68"

    Draw-RoundedBox $g 540 500 760 160 "#fffaf1" "#e9c46a"
    Draw-Label $g "Proposed Full-Stack Path" 565 530 20 "#7c4a03" $true
    Draw-Label $g "multimodal sensing -> synchronization -> transformer/task heads -> digital twin" 565 565 13 "#523f1b"
    Draw-Label $g "-> KG retrieval -> LLM explanation -> HITL dashboard -> telepresence and privacy-aware routing" 565 595 13 "#523f1b"

    Save-Canvas $canvas (Join-Path $FigureDir "system_architecture_overview.png")
}

function Generate-CaseStudyCards {
    for ($i = 1; $i -le 8; $i++) {
        $summary = Import-Csv (Join-Path $TableDir ("case_study_{0}_summary.csv" -f $i))
        $metrics = Import-Csv (Join-Path $CsvDir ("case_study_{0}_metrics.csv" -f $i))
        $row = $summary[0]

        $canvas = New-Canvas -Width 1200 -Height 520
        $g = $canvas.Graphics
        $statusColor = switch ($row.evidence_status) {
            "implemented_real_baseline plus planned extensions" { "#2a9d8f" }
            "simulation_based_evaluation" { "#e9c46a" }
            default { "#e76f51" }
        }

        Draw-RoundedBox $g 0 0 1200 520 "#ffffff" "#ffffff"
        Draw-Label $g ("Case Study CS{0}" -f $i) 55 35 18 "#52606d" $true
        Draw-Label $g $row.title 55 68 24 "#133c55" $true
        Draw-RoundedBox $g 55 115 290 44 $statusColor $statusColor
        Draw-Label $g $row.evidence_status 70 126 12 "#ffffff" $true

        Draw-RoundedBox $g 55 185 480 250 "#f8fafc" "#cbd2d9"
        Draw-Label $g "Existing assets" 80 210 18 "#133c55" $true
        Draw-Label $g $row.existing_assets 80 245 13 "#334e68"
        Draw-Label $g "Baselines: $($row.baselines)" 80 290 13 "#334e68" $true
        Draw-Label $g "Expected output" 80 335 16 "#133c55" $true
        Draw-Label $g $row.expected_output 80 365 13 "#334e68"

        Draw-RoundedBox $g 575 185 570 250 "#fffaf1" "#e9c46a"
        Draw-Label $g "Metric plan" 600 210 18 "#7c4a03" $true
        for ($m = 0; $m -lt [Math]::Min($metrics.Count, 5); $m++) {
            $metric = $metrics[$m]
            Draw-Label $g ("- {0} ({1})" -f $metric.metric_name, $metric.objective_direction) 600 (245 + 35 * $m) 12 "#523f1b"
        }

        Save-Canvas $canvas (Join-Path $FigureDir ("case_study_{0}_design_matrix.png" -f $i))
    }
}

function Generate-CaseStudyDashboard {
    $rows = Import-Csv (Join-Path $CsvDir "case_study_registry.csv")
    $implemented = ($rows | Where-Object { $_.evidence_status -eq "implemented_real_baseline plus planned extensions" }).Count
    $simulation = ($rows | Where-Object { $_.evidence_status -eq "simulation_based_evaluation" }).Count
    $planned = ($rows | Where-Object { $_.evidence_status -eq "planned_experiment" }).Count

    $canvas = New-Canvas -Width 1100 -Height 500
    $g = $canvas.Graphics
    Draw-Label $g "Case Study Evidence Distribution" 55 35 24 "#133c55" $true
    Draw-Label $g "Source: outputs/csv/case_study_registry.csv" 55 70 12 "#52606d"

    $items = @(
        @{ Label = "Implemented+Planned"; Count = $implemented; Color = "#2a9d8f"; X = 180 },
        @{ Label = "Simulation"; Count = $simulation; Color = "#e9c46a"; X = 460 },
        @{ Label = "Planned"; Count = $planned; Color = "#e76f51"; X = 740 }
    )

    foreach ($item in $items) {
        $height = $item.Count * 70
        $y = 420 - $height
        Draw-RoundedBox $g $item.X $y 110 $height $item.Color $item.Color
        Draw-Label $g "$($item.Count)" ($item.X + 40) ($y - 30) 16 "#1f2933" $true
        Draw-Label $g $item.Label ($item.X - 10) 435 12 "#334e68" $true
    }

    Save-Canvas $canvas (Join-Path $FigureDir "case_study_summary_dashboard.png")
}

Generate-LiteratureHeatmap
Generate-LiteratureBar
Generate-ArchitectureFigure
Generate-CaseStudyCards
Generate-CaseStudyDashboard
