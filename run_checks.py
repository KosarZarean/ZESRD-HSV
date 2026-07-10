#!/usr/bin/env python3
"""
Automated Test Runner and Fix System
Runs check_pipeline.py iteratively and applies fixes based on errors.
"""

import subprocess
import sys
import re
from pathlib import Path

def run_check_pipeline():
    """Run the check_pipeline.py script and capture output."""
    print("\n" + "="*80)
    print("RUNNING: python check_pipeline.py")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, 'check_pipeline.py'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
        
        # Print all output
        if stdout:
            print(stdout)
        if stderr:
            print("STDERR:", stderr)
        
        return returncode, stdout, stderr
        
    except subprocess.TimeoutExpired:
        print("ERROR: Script execution timed out (60s)")
        return -1, "", "Timeout"
    except Exception as e:
        print(f"ERROR: Failed to run script: {str(e)}")
        return -1, "", str(e)

def analyze_error(stdout, stderr):
    """Analyze error messages and return root cause."""
    combined = stdout + "\n" + stderr
    
    # Check for specific errors
    if "ModuleNotFoundError" in combined or "ImportError" in combined:
        match = re.search(r"(ModuleNotFoundError|ImportError): (.+)", combined)
        if match:
            return "import_error", match.group(2)
    
    if "AttributeError" in combined:
        match = re.search(r"AttributeError: (.+)", combined)
        if match:
            return "attribute_error", match.group(1)
    
    if "NameError" in combined:
        match = re.search(r"NameError: (.+)", combined)
        if match:
            return "name_error", match.group(1)
    
    if "TypeError" in combined:
        match = re.search(r"TypeError: (.+)", combined)
        if match:
            return "type_error", match.group(1)
    
    if "RuntimeError" in combined:
        match = re.search(r"RuntimeError: (.+)", combined)
        if match:
            return "runtime_error", match.group(1)
    
    if "Shape mismatch" in combined or "shape" in combined.lower():
        return "shape_error", "Tensor dimension mismatch detected"
    
    if "NaN" in combined:
        return "nan_error", "NaN value detected in computation"
    
    if "out of memory" in combined.lower():
        return "memory_error", "Out of memory error"
    
    return "unknown_error", "Unknown error occurred"

def main():
    """Main execution loop."""
    print("="*80)
    print("ZESRD-HSV PIPELINE VERIFICATION - AUTOMATED TEST RUNNER")
    print("="*80)
    
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}/{max_iterations}")
        print(f"{'='*80}")
        
        returncode, stdout, stderr = run_check_pipeline()
        
        if returncode == 0:
            print("\n" + "="*80)
            print("SUCCESS! Pipeline verification completed without errors.")
            print("="*80)
            return 0
        
        error_type, error_msg = analyze_error(stdout, stderr)
        print(f"\n[ERROR ANALYSIS] Type: {error_type}")
        print(f"[ERROR ANALYSIS] Message: {error_msg}")
        
        if iteration >= max_iterations:
            print(f"\nMax iterations ({max_iterations}) reached. Unable to auto-fix.")
            print("Please review errors above manually.")
            return 1
        
        print(f"\nPreparing iteration {iteration + 1}...")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
