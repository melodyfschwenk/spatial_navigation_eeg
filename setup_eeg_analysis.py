#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup Script for BrainVision EEG Analysis Environment
=====================================================

This script checks for and installs all necessary packages for BrainVision EEG data analysis.
It will:
1. Check existing installations
2. Install missing packages
3. Verify specific versions needed for compatibility
4. Print a summary of installed packages
"""

import sys
import subprocess
import importlib
import os
import platform
from packaging import version

# Define required packages with minimum versions
REQUIRED_PACKAGES = {
    'numpy': '1.20.0',
    'pandas': '1.3.0',
    'matplotlib': '3.4.0',
    'seaborn': '0.11.0',
    'scipy': '1.7.0',
    'mne': '0.24.0',      # MNE for EEG processing
    'statsmodels': '0.13.0',
    'scikit-learn': '1.0.0',  # For machine learning components
}

# Define optional packages that enhance functionality
OPTIONAL_PACKAGES = {
    'pyvistaqt': '0.4.0',  # For 3D visualization in MNE
    'tqdm': '4.60.0',      # Progress bars
    'pyEDFlib': '0.1.22',  # For EDF format support (if needed)
}

def check_package(package_name, min_version=None):
    """Check if package is installed with minimum version"""
    try:
        pkg = importlib.import_module(package_name)
        if hasattr(pkg, '__version__'):
            current_version = pkg.__version__
        elif hasattr(pkg, 'version'):
            current_version = pkg.version.__version__
        else:
            # Some packages store version info differently
            try:
                current_version = pkg.VERSION
            except:
                current_version = "Unknown"
        
        print(f"✓ {package_name} found (version: {current_version})")
        
        if min_version and current_version != "Unknown":
            try:
                if version.parse(current_version) < version.parse(min_version):
                    print(f"  ⚠ Warning: {package_name} version {current_version} is older than recommended {min_version}")
                    return False
            except:
                print(f"  ⚠ Warning: Could not compare versions for {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} not found")
        return False

def install_package(package_name, min_version=None):
    """Install package with pip"""
    package_spec = package_name
    if min_version:
        package_spec = f"{package_name}>={min_version}"
    
    print(f"Installing {package_spec}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_spec])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_name}: {e}")
        return False

def check_environment():
    """Check the Python environment"""
    print("\n=== Python Environment ===")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    # Check if pip is available
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], stdout=subprocess.PIPE)
        print("✓ pip is available")
    except:
        print("✗ pip is not available. Please install pip to continue.")
        return False
    
    # Check for virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print("✓ Running in a virtual environment")
    else:
        print("⚠ Not running in a virtual environment - consider creating one")
    
    return True

def print_section_header(title):
    """Print a formatted section header"""
    print(f"\n=== {title} ===")

def main():
    """Main installation process"""
    print("\nBrainVision EEG Analysis Environment Setup")
    print("==========================================")
    
    # Check environment
    if not check_environment():
        print("Environment check failed. Please fix the issues and try again.")
        return 1
    
    # First check if packaging is available for version comparisons
    try:
        import packaging
    except ImportError:
        print("\nInstalling 'packaging' for version comparisons...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging"])
    
    # Check required packages
    print_section_header("Required Packages")
    packages_to_install = []
    
    for package, min_version in REQUIRED_PACKAGES.items():
        if not check_package(package, min_version):
            packages_to_install.append((package, min_version))
    
    # Check optional packages
    print_section_header("Optional Packages")
    optional_to_install = []
    
    for package, min_version in OPTIONAL_PACKAGES.items():
        if not check_package(package, min_version):
            optional_to_install.append((package, min_version))
    
    # Install required missing packages
    if packages_to_install:
        print_section_header("Installing Required Packages")
        for package, min_version in packages_to_install:
            install_package(package, min_version)
    else:
        print("\nAll required packages are already installed!")
    
    # Offer to install optional packages
    if optional_to_install:
        print_section_header("Optional Packages Installation")
        print("The following optional packages enhance functionality but aren't required:")
        for package, min_version in optional_to_install:
            print(f"- {package} (>= {min_version})")
        
        response = input("\nDo you want to install these optional packages? (y/n): ")
        if response.lower() in ('y', 'yes'):
            for package, min_version in optional_to_install:
                install_package(package, min_version)
    
    # Special MNE installation instruction for nicer 3D visualizations
    print_section_header("MNE Additional Setup")
    try:
        import mne
        has_pyvista = False
        try:
            import pyvistaqt
            has_pyvista = True
        except ImportError:
            pass
        
        if not has_pyvista:
            print("For 3D visualization in MNE, additional packages are recommended.")
            print("Run the following command separately if you need 3D plotting:")
            print("pip install pyvistaqt PyQt5")
    except ImportError:
        # MNE should have been installed already if missing
        pass
    
    # Verify BrainVision support in MNE
    print_section_header("BrainVision Support Verification")
    try:
        import mne
        print("Checking MNE BrainVision reader functionality...")
        has_brainvision = hasattr(mne.io, 'read_raw_brainvision')
        if has_brainvision:
            print("✓ MNE BrainVision reader is available")
        else:
            print("✗ MNE appears to be missing BrainVision reader support")
    except ImportError:
        print("Cannot verify BrainVision support - MNE import failed")
    
    # Final instructions
    print_section_header("Setup Complete")
    print("Environment setup is complete. You can now run the BrainVision analysis scripts.")
    print("\nSuggested next steps:")
    print("1. Run the example script:")
    print("   python analysis/brainvision_example.py")
    print("2. Create your own condition mapping based on your experimental design")
    print("3. Extract epochs and perform your intended analyses")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())