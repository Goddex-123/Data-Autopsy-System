#!/usr/bin/env python3
"""
Data Autopsy System - Command Line Interface

Forensic analysis of datasets to uncover bias, manipulation, and misleading patterns.
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autopsy import DataAutopsy


def main():
    parser = argparse.ArgumentParser(
        description='🔬 Data Autopsy: Forensic Dataset Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data.csv
  python main.py --input data.csv --output reports/
  python main.py --input data.csv --format html
  python main.py --input data.csv --quick
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to the input dataset (CSV file)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='output',
        help='Output directory for reports and visualizations (default: output)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['html', 'md', 'json', 'all'],
        default='html',
        help='Report format (default: html)'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick scan only (no full analysis)'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
        
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  🔬 DATA AUTOPSY SYSTEM                      ║
║              Forensic Dataset Analysis Tool                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"📂 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    print()
    
    try:
        # Initialize autopsy
        autopsy = DataAutopsy(args.input)
        
        if args.quick:
            # Quick scan only
            print("🔍 Running quick scan...")
            results = autopsy.quick_scan()
            
            print("\n📊 Quick Scan Results:")
            print("-" * 40)
            for key, value in results.items():
                print(f"  {key}: {value}")
            print()
            
        else:
            # Full forensic investigation
            print("🔬 Starting full forensic investigation...")
            print("   This may take a moment for large datasets.\n")
            
            report = autopsy.investigate(output_dir=args.output)
            
            # Print summary
            report.print_summary()
            
            # Save reports
            if args.format == 'all':
                report.save('forensic_report.html')
                report.save('forensic_report.md')
                report.save('forensic_report.json')
            else:
                report.save(f'forensic_report.{args.format}')
                
            print("\n✅ Analysis complete!")
            print(f"📁 Reports and visualizations saved to: {os.path.abspath(args.output)}")
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
