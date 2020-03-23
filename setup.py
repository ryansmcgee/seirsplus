import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    packages=setuptools.find_packages(),
    version='0.0.1',
    description='Models for SEIRS epidemic dynamics on networks and with interventions, such as testing, contact tracing, and social distancing.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryansmcgee/SEIRS-network-model",
    name="ryansmcgee",
    author='Ryan Seamus McGee',
    author_email='ryansmcgee@gmail.com',
    license='MIT',
    install_requires=['numpy', 'scipy', 'networkx'],
    zip_safe=False)