"""
Test script to verify that all required packages are properly installed
and provide a simple example to confirm they're working correctly.
"""

import os
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}\n")

# First check basic packages
print("=== Testing Basic Packages ===")
packages = {
    "numpy": "np",
    "pandas": "pd",
    "matplotlib.pyplot": "plt",
    "seaborn": "sns",
    "scipy": "scipy"
}

for package, alias in packages.items():
    try:
        exec(f"import {package} as {alias}")
        version = eval(f"{alias}.__version__")
        print(f"✓ {package} successfully imported (version: {version})")
    except ImportError as e:
        print(f"✗ Failed to import {package}: {e}")
    except AttributeError:
        print(f"✓ {package} successfully imported (version unknown)")

# Test MNE
print("\n=== Testing MNE ===")
try:
    import mne
    from mne.time_frequency import tfr_morlet
    print(f"✓ MNE successfully imported (version: {mne.__version__})")
    
    # Create some sample data to test MNE
    import numpy as np
    data = np.random.randn(3, 1000)  # 3 channels, 1000 timepoints
    info = mne.create_info(ch_names=['Fz', 'Cz', 'Pz'], sfreq=100, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    print("✓ Successfully created MNE Raw object")
except ImportError as e:
    print(f"✗ Failed to import MNE: {e}")
except Exception as e:
    print(f"✗ Error testing MNE functionality: {str(e)}")

# Test statsmodels
print("\n=== Testing statsmodels ===")
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import AnovaRM
    print(f"✓ statsmodels successfully imported (version: {sm.__version__})")
    
    # Simple regression test
    x = np.arange(10)
    y = x * 2 + np.random.normal(size=10)
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    print("✓ Successfully fit statsmodels OLS model")
    print(f"  R-squared: {results.rsquared:.4f}")
except ImportError as e:
    print(f"✗ Failed to import statsmodels: {e}")
except Exception as e:
    print(f"✗ Error testing statsmodels functionality: {str(e)}")

# Test scikit-learn
print("\n=== Testing scikit-learn ===")
try:
    import sklearn
    print(f"✓ scikit-learn successfully imported (version: {sklearn.__version__})")
    
    from sklearn.ensemble import RandomForestClassifier
    print("✓ Successfully imported RandomForestClassifier")
except ImportError as e:
    print(f"✗ Failed to import scikit-learn: {e}")
    
print("\n=== VS Code Configuration Tips ===")
print("If VS Code still shows import errors:")
print("1. Restart VS Code completely")
print("2. Make sure you've selected the correct Python interpreter:")
print("   Ctrl+Shift+P → Python: Select Interpreter → Choose the one used for installation")
print(f"3. Check that the .env file exists in your project directory")

# Print site-packages for reference
import site
print("\nYour site-packages directories:")
for path in site.getsitepackages():
    print(f"  {path}")

print("\nIf VS Code still can't find the modules, add these to your .env file:")
print("PYTHONPATH=" + os.pathsep.join(site.getsitepackages()))
