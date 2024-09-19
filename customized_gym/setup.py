from setuptools import setup, find_packages

setup(
    name='gym',  # Replace with your package name
    version='0.19.0',  # Replace with your package version
    description='A customized gym package',  # Replace with a short description
    long_description=open('README.md').read(),  # Optional: Include a README file
    long_description_content_type='text/markdown',  # Optional: Specify the format of the long description
    author='zhugb',  # Replace with your name
    author_email='zhugb@buaa.edu.cn',  # Replace with your email
    packages=find_packages(exclude=['tests', 'docs']),  # Automatically find package directories
    install_requires=[
        # List your package's dependencies here
        # e.g., 'numpy>=1.19.2', 'requests>=2.25.1'
    ],
    extras_require={
        'dev': [
            # Development dependencies
            'pytest>=6.0.0',  # For testing
            'flake8>=3.8.4',  # For linting
        ],
    },
    entry_points={
        'console_scripts': [
            # Define command-line scripts here
            # 'script_name=your_package.module:function',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your license if different
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the Python version
)

