# Data Autopsy System

🔬 **Forensic Analysis of Datasets** - Treat your data like a crime scene.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Data Autopsy System is a master-level forensic analysis toolkit that investigates datasets for:

- **Bias** - Sampling, selection, and measurement biases
- **Manipulation** - Data fabrication and anomalies
- **Missing Signals** - Intentional omissions and silent assumptions
- **Misleading Patterns** - Fragile conclusions and spurious correlations

Used in journalism, courts, and intelligence work.

## Features

| Module | Description |
|--------|-------------|
| 🔍 **Provenance Analyzer** | Track data origins, collection methods, temporal patterns |
| ⚖️ **Bias Detector** | Statistical tests for sampling and selection bias |
| 🚨 **Anomaly Detector** | Benford's Law, outliers, fabrication signatures |
| 🕳️ **Missing Data Forensics** | Detect intentional omissions and coverage gaps |
| 🧪 **Robustness Tester** | Sensitivity analysis, bootstrap tests, fragility scoring |
| 📊 **Visual Evidence** | Professional forensic visualizations |
| 📋 **Report Generator** | Executive-style forensic reports |

## Installation

```bash
# Clone the repository
git clone https://github.com/Goddex-123/Data-Autopsy-System.git
cd Data-Autopsy-System

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from autopsy import DataAutopsy

# Load your dataset
autopsy = DataAutopsy("your_data.csv")

# Run full forensic analysis
report = autopsy.investigate()

# Generate executive report
report.save("forensic_report.html")
```

## Command Line Usage

```bash
# Full autopsy on a dataset
python main.py --input data.csv --output report.html

# Run specific analysis
python main.py --input data.csv --analyze bias anomaly
```

## Example Output

Run the demo:

```bash
python examples/full_autopsy.py
```

This generates:
- Visual evidence in `output/`
- Executive summary report
- Detailed findings with risk scores

## Project Structure

```
Data-Autopsy-System/
├── autopsy/                 # Core analysis modules
│   ├── provenance.py       # Data origin analysis
│   ├── bias_detector.py    # Bias detection engine
│   ├── anomaly_detector.py # Anomaly detection
│   ├── missing_analyzer.py # Missing data forensics
│   ├── robustness.py       # Conclusion testing
│   └── visualizer.py       # Visual evidence
├── reports/                 # Report generation
├── examples/                # Demo scripts
├── datasets/                # Sample data
└── output/                  # Generated reports
```

## Use Cases

- **Investigative Journalism** - Verify data claims and sources
- **Legal Discovery** - Present data evidence in court
- **Academic Review** - Validate research datasets
- **Corporate Auditing** - Detect data manipulation
- **Intelligence Analysis** - Assess data reliability

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*"Data doesn't lie, but liars use data."* - This tool helps you find the truth.
