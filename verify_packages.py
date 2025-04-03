#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Package Verification and IDE Configuration Helper
================================================

This script verifies your installed packages are working correctly and 
helps configure your IDE to recognize them properly.
"""

import os
import sys
import subprocess
import importlib
import site
import platform

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")

def check_package(package_name):
    """Check if package is properly installed and return version info"""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version.__version__
        else:
            try:
                version = module.VERSION
            except:
                version = "Unknown"
                
        print(f"✓ {package_name} is correctly installed (version: {version})")
        return True, module.__file__, version
    except ImportError as e:
        print(f"✗ {package_name} import failed: {str(e)}")
        return False, None, None

def get_python_env_info():
    """Get information about the Python environment"""
    print_section("Python Environment Information")
    
    print(f"Python version: {platform.python_version()}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print(f"✓ Running in a virtual environment: {sys.prefix}")
    else:
        print("✗ Not running in a virtual environment")
    
    # Get site-packages directories
    print("\nSite-packages directories:")
    for path in site.getsitepackages():
        print(f"  - {path}")
    
    # Get user site-packages
    user_site = site.getusersitepackages()
    print(f"\nUser site-packages: {user_site}")
    
    # Get PYTHONPATH environment variable
    pythonpath = os.environ.get('PYTHONPATH', '')
    if pythonpath:
        print(f"\nPYTHONPATH environment variable:")
        for path in pythonpath.split(os.pathsep):
            print(f"  - {path}")
    else:
        print("\nPYTHONPATH environment variable is not set")

def test_mne_functionality():
    """Test basic MNE functionality"""
    print_section("Testing MNE Functionality")
    
    try:
        import mne
        import numpy as np
        
        print(f"MNE version: {mne.__version__}")
        
        # Create some test data
        print("\nCreating sample data...")
        data = np.random.randn(3, 100)  # 3 channels, 100 time points
        ch_names = ['F3', 'Fz', 'F4']
        ch_types = ['eeg', 'eeg', 'eeg']
        sfreq = 100  # 100 Hz sampling frequency
        
        # Create info object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # Create raw object
        raw = mne.io.RawArray(data, info)
        print("✓ Successfully created MNE Raw object")
        
        # Test epochs creation
        events = np.array([[10, 0, 1], [50, 0, 2]])
        epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.3, baseline=(None, 0))
        print("✓ Successfully created MNE Epochs object")
        
        return True
    except Exception as e:
        print(f"✗ Error testing MNE functionality: {str(e)}")
        return False

def test_statsmodels_functionality():
    """Test basic statsmodels functionality"""
    print_section("Testing statsmodels Functionality")
    
    try:
        import statsmodels.api as sm
        import numpy as np
        import pandas as pd
        
        print(f"statsmodels version: {sm.__version__}")
        
        # Create some test data
        print("\nCreating sample data...")
        np.random.seed(9876789)
        x = np.random.randn(100)
        X = sm.add_constant(x)
        beta = [1, 0.5]
        e = np.random.randn(100) * 0.5
        y = np.dot(X, beta) + e
        
        # Fit OLS model
        model = sm.OLS(y, X)
        results = model.fit()
        print("✓ Successfully fitted OLS model")
        print(f"  - R-squared: {results.rsquared:.4f}")
        print(f"  - Parameters: {results.params}")
        
        # Test ANOVA
        try:
            from statsmodels.formula.api import ols
            from statsmodels.stats.anova import anova_lm
            
            # Create a test DataFrame
            df = pd.DataFrame({
                'value': np.random.randn(30), 
                'group': np.repeat(['A', 'B', 'C'], 10)
            })
            
            # Fit model and run ANOVA
            model = ols('value ~ group', data=df).fit()
            anova_table = anova_lm(model)
            print("✓ Successfully ran ANOVA")
            
        except Exception as e:
            print(f"✗ Error testing ANOVA: {str(e)}")
        
        return True
    except Exception as e:
        print(f"✗ Error testing statsmodels functionality: {str(e)}")
        return False

def get_ide_config_recommendations():
    """Provide recommendations for configuring IDEs"""
    print_section("IDE Configuration Recommendations")
    
    # VS Code recommendations
    print("Visual Studio Code:")
    print("1. Make sure to select the correct Python interpreter:")
    print("   - Press Ctrl+Shift+P and type 'Python: Select Interpreter'")
    print("   - Select the interpreter that contains your installed packages")
    print(f"   - Current interpreter: {sys.executable}")
    print("\n2. Check your settings.json:")
    print('   - Add: "python.analysis.extraPaths": ["path/to/site-packages"]')
    print("\n3. Try creating a .env file in your workspace with:")
    print(f'   PYTHONPATH={os.pathsep.join(site.getsitepackages())}')
    
    # PyCharm recommendations
    print("\nPyCharm:")
    print("1. Go to File > Settings > Project > Python Interpreter")
    print("2. Ensure the correct interpreter is selected")
    print("3. Check that packages appear in the package list")
    print("4. If using Scientific Mode, try restarting the console")
    
    # General recommendations
    print("\nGeneral tips:")
    print("1. Try restarting your IDE")
    print("2. If using a virtual environment, make sure it's activated")
    print("3. Check for conflicting package versions")
    print("4. Verify packages are installed in the correct Python environment")
    print("   (the one your IDE is using)")

def main():
    """Main function to run package verification"""
    print("\nPACKAGE VERIFICATION AND IDE CONFIGURATION HELPER")
    print("===============================================")
    
    # Get Python environment info
    get_python_env_info()
    
    # Check required packages
    print_section("Checking Required Packages")
    packages_to_check = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy',
        'mne', 'statsmodels', 'scikit-learn'
    ]
    
    package_results = {}
    for package in packages_to_check:
        success, path, version = check_package(package)
        package_results[package] = {'success': success, 'path': path, 'version': version}
    
    # Additional checks for problematic packages
    mne_success = False
    if package_results['mne']['success']:
        # Check MNE submodules
        try:
            from mne.time_frequency import tfr_morlet
            print("✓ mne.time_frequency.tfr_morlet is accessible")
            mne_success = True
        except ImportError as e:
            print(f"✗ Error importing mne.time_frequency.tfr_morlet: {str(e)}")
    
    statsmodels_success = False
    if package_results['statsmodels']['success']:
        # Check statsmodels submodules
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            from statsmodels.stats.anova import AnovaRM
            print("✓ All statsmodels submodules are accessible")
            statsmodels_success = True
        except ImportError as e:
            print(f"✗ Error importing statsmodels submodules: {str(e)}")
    
    # Run functionality tests
    if mne_success:
        test_mne_functionality()
    
    if statsmodels_success:
        test_statsmodels_functionality()
    
    # Provide IDE configuration recommendations
    get_ide_config_recommendations()
    
    # Summarize results
    print_section("Summary")
    all_ok = all(result['success'] for result in package_results.values())
    
    if all_ok:
        print("✓ All required packages are correctly installed!")
        print("\nIf your IDE still shows import errors:")
        print("1. Follow the IDE configuration recommendations above")
        print("2. Try adding this to the beginning of your scripts:")
        print("\nimport sys")
        for path in site.getsitepackages():
            print(f"sys.path.append(r'{path}')")
        
    else:
        failed_packages = [pkg for pkg, result in package_results.items() if not result['success']]
        print(f"✗ The following packages had issues: {', '.join(failed_packages)}")
        print("\nPlease check the specific error messages above.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())