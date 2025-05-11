from setuptools import setup, find_packages

setup(
    name="ma_agent_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'yfinance',
        'pandas',
        'numpy',
        'ta',
        'python-dotenv',
    ],
) 