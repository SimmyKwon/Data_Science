from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

# Additional installation for Playwright
class CustomInstallCommand(install):
    def run(self):
        # 1. Run the installation codes on setup
        super().run()
        # 2. Install Playwright binaries
        print("📥 Downloading Playwright Browsers...")
        subprocess.check_call(["playwright", "install", "chromium"]) # Download only what is needed
setup(
    name="phishing-detector-api",
    version="1.0.0",
    author="SimmyK",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.136.1",
        "uvicorn[standard]==0.46.0",
        "playwright==1.59.0",
        "joblib==1.5.3",
        "pandas==3.0.2",
        "scikit-learn==1.8.0",
    ],
    
    python_requires=">=3.11.15",

    # Use CustomInstallCommand after setting up the required packages.
    cmdclass={
        'install': CustomInstallCommand,
    },
)